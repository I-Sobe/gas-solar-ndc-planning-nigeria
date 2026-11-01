"""
99c_run_genset_arm_sensitivity.py  --  Genset arm sensitivity
==============================================================

REPLACES 99c_run_voll_sensitivity.py (correction plan 2.6, F1a).

WHY THE OLD SCRIPT WAS RETIRED
------------------------------
Its premise was: sweep VoLL to find the level at which the grid service
constraint transitions from non-binding to binding. With a self-generation
backstop that transition can no longer occur. selfgen absorbs the shortfall
before VoLL is ever reached, so `unserved` is zero at every VoLL level and the
sweep measures nothing.

Worse, the old script reported `unserved` but not `selfgen`, and it loaded econ
without a genset_case -- so selfgen sat at the VoLL fallback, the split between
the two was degenerate, and the LP silently routed the entire shortfall into a
variable the script never printed. It produced `unserved = 0.00` across the
board while real shortfall of hundreds of TWh went unreported.

WHAT THIS DOES INSTEAD
----------------------
Sweeps the three genset pricing arms -- the sensitivity that the three-arm
structure was built to provide, and which replaces the deleted VoLL sweep:

    genset_srmc     eta 0.3474 (75% load)     277 USD/MWh   EF_e 0.729
    genset_central  eta 0.2809 (25% load)     445 USD/MWh   EF_e 0.902
    genset_high     eta 0.0511 (A2EI metered) 3323 USD/MWh  EF_e 4.638

crossed with the two financing arms (public_only, eaas) and a small set of
grid service obligations calibrated to the frontier (0.30 was the tightest
feasible under tight capital without an emissions cap; with the NDC3 cap on top
it may be tighter, so infeasibility is recorded rather than raised).

WHAT TO READ
------------
1. Does the public-vs-EaaS conclusion survive the choice of genset arm? If EaaS
   wins under all three, the finding is robust to the single most uncertain
   parameter in 2.6.
2. Where does public_only go infeasible while eaas remains feasible? That is a
   feasibility-frontier result, not a cost result, and it is stronger.
3. How much of total emissions is selfgen under each arm? Sizes the leakage
   exposure ahead of F3.

CAUTION ON unserved
-------------------
Under any genset pricing, `unserved` is near zero BY CONSTRUCTION -- the
backstop is cheaper and absorbs the shortfall. It is NOT evidence that demand
is met. Read `shortfall_twh` (= selfgen + unserved) as the service shortfall,
and the selfgen cost/CO2 as the burden it imposes on private actors.

PRE-REQUISITES
--------------
    python scripts/00_build_emissions_cap.py
"""

import argparse
import sys
from pathlib import Path

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

RESULTS_DIR = ROOT / "results" / "genset_arm_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"

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
# NOTE (TASK 29): the FINANCING_ARMS entries formerly carried
# required_margin=1.10 on BOTH arms; that literal (the pre-F1 fork
# dscr_scenarios() retired, applied on the EaaS arm where it is not inert) is
# removed. required_margin is the dscr_case-derived value load_scenario() sets.
CAPITAL_RECOVERY_CASE_CHOICES = ("cr_central", "cr_ida21", "cr_comm_low", "cr_comm_high")
DSCR_CASE_CHOICES = ("dscr_regression", "dscr_central", "dscr_low", "dscr_high")
CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")
# Envelope resolution: F1's own concessional envelope arm, REQUIRED, no
# default -- see concessional_envelope_scenarios() in src/scenarios.py.
CONCESSIONAL_ENVELOPE_CASE_CHOICES = ("env_dares", "env_dares_plus", "env_none", "env_5x", "env_10x")


def parse_args():
    parser = argparse.ArgumentParser(description="99c_run_genset_arm_sensitivity: genset pricing arm sensitivity")
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
    # missing before this edit). tariff_case is inert for the public_only
    # FINANCING_ARMS entry but read for the eaas entry (m.eaas_disable does
    # NOT fix solar_eaas_add under financing_regime="eaas") -- unlike 24_/
    # 25_, this runner's eaas arm makes tariff_case load-bearing, not inert.
    add_standard_arms(parser, include=["conversion_case", "hydro_case", "tariff_case", "om_case"])
    return parser.parse_args()

GENSET_ARMS = ["genset_srmc", "genset_central", "genset_high"]
# The unconstrained optimum shortfall is ~0.31, so anything above it is
# interior and the sweep traces nothing. The frontier lives BELOW 0.31.
EPS_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
GRID_SERVICE_MODE = "annual"   # 'total' lets early slack subsidise late
                               # collapse under a rising deficit; see 2.6E.
NDC_SCENARIO = "ndc3_unconditional"

FINANCING_ARMS = [
    {"label": "public_only", "financing_regime": "public", "capital_case": "tight"},
    {"label": "eaas", "financing_regime": "eaas", "capital_case": "tight"},
]


def load_annual_caps(scenario_name, years):
    if not CAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CAP_PATH}. Run scripts/00_build_emissions_cap.py first."
        )
    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].sort_values("year")
    df = df[df["year"].isin([int(y) for y in years])].sort_values("year")
    caps = df["cap_tco2"].astype(float).tolist()
    if len(caps) != len(years):
        raise ValueError(
            f"Cap length {len(caps)} != model years {len(years)} for "
            f"scenario={scenario_name}."
        )
    return caps


def solve_one(genset_arm, fin_arm, eps, peak_case, storage_cc_case,
              gas_build_case, new_gas_deliverability, gas_availability_case,
              gas_retirement_case, gas_capex_case, ccgt_efficiency_case,
              capital_recovery_case, dscr_case, us_inflation, capital_treatment_case,
              concessional_envelope_case, conversion_case, hydro_case,
              tariff_case, om_case):
    sc = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        gas_deliverability_case="baseline",
        capital_case=fin_arm["capital_case"],
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
    sc["financing_regime"] = fin_arm["financing_regime"]
    # TASK 29: no required_margin override (dscr_case-derived; register L34).

    econ = load_econ("voll_mid", genset_case=genset_arm)
    years = sc["years"]
    caps = load_annual_caps(NDC_SCENARIO, years)
    capex = load_solar_capex_by_year("solar_low", int(years[0]), int(years[-1]))

    base = dict(genset_arm=genset_arm, financing_arm=fin_arm["label"], eps=eps,
                peak_case=peak_case, storage_cc_case=storage_cc_case,
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
                om_case=om_case)

    try:
        m = build_model_sliced(
            scenario=sc,
            econ=econ,
            emissions_cap_by_year=caps,
            max_offgrid_fraction=eps,
            grid_service_mode=GRID_SERVICE_MODE,
            solar_capex_by_year=capex,
        )
        status = solve_model(m, scenario=sc)
        if not status.get("optimal", False):
            return {**base, "feasible": False, "reason": "not optimal"}
    except RuntimeError as exc:
        # Infeasibility is a DATA POINT: public_only failing where eaas
        # succeeds is a feasibility-frontier result, not an error.
        return {**base, "feasible": False, "reason": str(exc)[:100]}

    n = len(years)
    # Sliced port (A5, Group B): m.selfgen/m.unserved/m.balance are all
    # [T, m.S]-indexed under build_model_sliced, not [T]. Summing over m.S
    # reconstructs the same annual totals the [t]-only reads gave before.
    #
    # m.balance[t].lower specifically: reported before rewriting, per A5's
    # own instruction, since a constraint's .lower is not automatically a
    # sum. Investigated first: m.balance's rule is
    # "<generation terms> >= demand[(t,s,p)]" (optimize_model_sliced.py),
    # so .lower on a given (t,(s,p)) key returns exactly that slice's fixed
    # demand DATA -- not a solved/dual value, and not something a slice
    # boundary changes the meaning of. Demand is additive across the six
    # slices by construction (they partition the year's 8760 hours
    # exactly), so summing .lower over m.S gives the SAME annual total
    # demand the single annual m.balance[t].lower gave before. Verified
    # directly against a live sliced solve before this edit: t=0 summed to
    # 37.09 (TWh), matching scenario["base_demand_twh"] exactly. Meaning
    # preserved.
    selfgen = sum(float(pyo.value(m.selfgen[t, (s, p)])) for t in range(n) for (s, p) in m.S)
    unserved = sum(float(pyo.value(m.unserved[t, (s, p)])) for t in range(n) for (s, p) in m.S)
    demand = sum(float(pyo.value(m.balance[t, (s, p)].lower)) for t in range(n) for (s, p) in m.S)
    # Leakage guard (2.6, D15). Cap is grid-only; if load is dumped to diesel
    # the NDC is "met" on paper while national emissions rise. A flagged run
    # is not an NDC-compliant plan and must not be reported as one.
    leak = leakage_diagnostic(m)
    if leak["flagged"]:
        print(f"  LEAKAGE FLAG [{genset_arm}/{fin_arm['label']} eps={eps}]: "
              f"{leak['leakage_share']:.1%} of emissions "
              f"({leak['leakage_tco2']/1e6:.1f} MtCO2) outside the cap boundary. "
              f"NOT NDC-compliant.")
        
    return {
        **base,
        "feasible": True,
        "objective_usd": float(pyo.value(m.system_cost_npv)),
        "selfgen_twh": selfgen,
        "unserved_twh": unserved,
        "shortfall_twh": selfgen + unserved,
        "shortfall_fraction": (selfgen + unserved) / demand if demand else None,
        "private_selfgen_npv_usd": float(pyo.value(m.cost_selfgen_npv)),
        "unserved_voll_npv_usd": float(pyo.value(m.cost_unserved_voll_npv)),
        "solar_built_mw": sum(
            float(pyo.value(m.solar_public_add[t])) + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(n)
        ),
        "storage_mwh": float(pyo.value(m.storage_capacity_mwh[n - 1])),
        "emissions_total_tco2": float(pyo.value(m.emissions)),
        "emissions_grid_only_tco2": float(pyo.value(m.emissions_grid_only)),
        "emissions_selfgen_tco2": float(pyo.value(m.emissions_selfgen)),
        **annual_emissions_flat(m, years),
        "leakage_share": leak["leakage_share"],
        "leakage_tco2": leak["leakage_tco2"],
        "leakage_flagged": leak["flagged"],
    }


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    rows = []
    total = len(GENSET_ARMS) * len(FINANCING_ARMS) * len(EPS_LEVELS)
    print(f"\nGenset arm sensitivity: {len(GENSET_ARMS)} arms x "
          f"{len(FINANCING_ARMS)} financing x {len(EPS_LEVELS)} eps = {total} solves")
    print(f"NDC scenario: {NDC_SCENARIO}   mode: {GRID_SERVICE_MODE}   "
          f"peak_case: {args.peak_case}   storage_cc_case: {args.storage_cc_case}   "
          f"gas_build_case: {args.gas_build_case}   "
          f"new_gas_deliverability: {args.new_gas_deliverability}   "
          f"gas_availability_case: {args.gas_availability_case}   "
          f"gas_retirement_case: {args.gas_retirement_case}   "
          f"gas_capex_case: {args.gas_capex_case}   "
          f"ccgt_efficiency_case: {args.ccgt_efficiency_case}\n")

    for arm in GENSET_ARMS:
        for fin in FINANCING_ARMS:
            for eps in EPS_LEVELS:
                rows.append(solve_one(arm, fin, eps, args.peak_case, args.storage_cc_case,
                                       args.gas_build_case, args.new_gas_deliverability,
                                       args.gas_availability_case, args.gas_retirement_case,
                                       args.gas_capex_case, args.ccgt_efficiency_case,
                                       args.capital_recovery_case, args.dscr_case,
                                       args.us_inflation, args.capital_treatment_case,
                                       args.concessional_envelope_case, args.conversion_case,
                                       args.hydro_case, args.tariff_case, args.om_case))

    tag = f"{args.peak_case}__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}__{args.gas_availability_case}__{args.gas_retirement_case}__{args.gas_capex_case}__{args.ccgt_efficiency_case}__{args.capital_recovery_case}__{args.dscr_case}__{args.capital_treatment_case}__{args.concessional_envelope_case}"
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"genset_arm_sensitivity__{tag}.csv", index=False)

    # ---- main table --------------------------------------------------------
    print(f"{'=' * 92}")
    print("  GENSET ARM SENSITIVITY")
    print(f"{'=' * 92}")
    print(f"  {'arm':>15}{'financing':>13}{'eps':>7}{'obj bn':>11}"
          f"{'short TWh':>12}{'selfgen bn':>12}{'sg MtCO2':>11}{'solar GW':>10}")
    for r in rows:
        if not r["feasible"]:
            print(f"  {r['genset_arm']:>15}{r['financing_arm']:>13}{r['eps']:>7.2f}"
                  f"{'INFEASIBLE':>56}")
            continue
        print(f"  {r['genset_arm']:>15}{r['financing_arm']:>13}{r['eps']:>7.2f}"
              f"{r['objective_usd'] / 1e9:>11.2f}"
              f"{r['shortfall_twh']:>12.1f}"
              f"{r['private_selfgen_npv_usd'] / 1e9:>12.2f}"
              f"{r['emissions_selfgen_tco2'] / 1e6:>11.1f}"
              f"{r['solar_built_mw'] / 1e3:>10.2f}")

    # ---- public vs EaaS, per arm ------------------------------------------
    print(f"\n{'=' * 92}")
    print("  PUBLIC vs EaaS -- does the conclusion survive the genset arm?")
    print(f"{'=' * 92}")
    print(f"  {'arm':>15}{'eps':>7}{'public bn':>13}{'eaas bn':>12}"
          f"{'saving bn':>13}{'saving %':>11}")
    for arm in GENSET_ARMS:
        for eps in EPS_LEVELS:
            pub = next((r for r in rows if r["genset_arm"] == arm
                        and r["financing_arm"] == "public_only" and r["eps"] == eps), None)
            eaas = next((r for r in rows if r["genset_arm"] == arm
                         and r["financing_arm"] == "eaas" and r["eps"] == eps), None)
            if not pub or not eaas:
                continue
            if not pub["feasible"] and eaas["feasible"]:
                print(f"  {arm:>15}{eps:>7.2f}{'INFEASIBLE':>13}"
                      f"{eaas['objective_usd'] / 1e9:>12.2f}"
                      f"{'--':>13}{'EaaS ONLY':>11}")
                continue
            if not pub["feasible"] or not eaas["feasible"]:
                print(f"  {arm:>15}{eps:>7.2f}{'both/neither infeasible':>49}")
                continue
            s = pub["objective_usd"] - eaas["objective_usd"]
            print(f"  {arm:>15}{eps:>7.2f}"
                  f"{pub['objective_usd'] / 1e9:>13.2f}"
                  f"{eaas['objective_usd'] / 1e9:>12.2f}"
                  f"{s / 1e9:>13.2f}"
                  f"{s / pub['objective_usd'] * 100:>10.1f}%")

    print("\n  'EaaS ONLY' rows are the strongest result available here: the")
    print("  obligation is unreachable on public financing alone but reachable")
    print("  with EaaS. That is a feasibility-frontier finding, not a cost one.")
    print("\n  CAUTION: 'unserved' is near zero by construction under every genset")
    print("  arm -- the backstop absorbs the shortfall. Quote 'short TWh'.")
    print(f"\n  saved: {RESULTS_DIR / f'genset_arm_sensitivity__{tag}.csv'}\n")


if __name__ == "__main__":
    main()
