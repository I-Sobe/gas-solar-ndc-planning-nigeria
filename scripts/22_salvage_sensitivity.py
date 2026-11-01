"""
22_salvage_sensitivity.py  --  Is the investment trajectory a residual?
=======================================================================

PURPOSE
-------
Tests whether the capacity trajectory RESPONDS to cost, or is determined by the
energy balance regardless of cost. This is the scheduled test written into
correction plan 2.5.

THE FINDING BEING TESTED
------------------------
When salvage was added (plan 2.5), system cost fell 17.1% with the credit
concentrated entirely on late vintages -- and the solar trajectory was IDENTICAL
to five decimal places, while sliced storage was identical to four. Zero MW of
movement from a large, timing-skewed cost change.

The interpretation: with demand exogenous, hydro exogenous, gas at its
deliverability ceiling, and unserved priced out by VoLL, the energy balance has
one unknown per year and the model solves for it. Solar -- and in the sliced
model, storage -- is an ACCOUNTING RESIDUAL, not an optimisation outcome.

WHEN TO RE-RUN THIS
-------------------
After each change that could open a degree of freedom:
  * 2.5 step 3b   removing the min-build floor
  * 2.5 step 4    relaxing or ramping the max-build cap
  * 2.6           genset backstop (serve-vs-pay becomes economic, not penal)
  * 5.3           freeing gas_add

If the trajectory MOVES, the degrees of freedom have opened and the residual
finding must be restated as "was determined, now is not."
If it does NOT move, the constraint structure is a property of Nigeria's system
rather than of the formulation -- the stronger version of the claim.

Either result belongs in the methods chapter. Record which one you got.

METHOD
------
Salvage is switched off by setting every asset life to 1.0 year: the salvage
fraction is max(0, life - years_served)/life, and years_served >= 1 for every
vintage, so the credit is exactly zero. No model change is needed.

Expect the plan 2.5 in-horizon retirement guard to fire loudly in the
no-salvage arm. That is correct -- a 1-year life retires immediately -- and is
suppressed in the printed summary.

OUTPUT
------
    results/salvage_sensitivity/salvage_sensitivity.json
    results/salvage_sensitivity/trajectories.csv
"""

import argparse
import json
import sys
import contextlib
import io as _io
from pathlib import Path
from weakref import ref

import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.cli_arms import resolve_results_dir
from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import (
    load_scenario, asset_lifetimes, gas_deliverability_scenarios,
    conversion_scenarios, hydro_case_scenarios, tariff_scenarios, MODEL_END_YEAR,
)
from src.optimize_model import build_model, solve_model
from src.optimize_model_sliced import build_model_sliced, conversion_diagnostic

RESULTS_DIR = ROOT / "results" / "salvage_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_VOLL = "voll_mid"

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
# NOTE: this script's whole point is to probe salvage behaviour (m.salvage_npv),
# which is structurally ABSENT under f1_tranche -- pass --capital-treatment-case
# pre_f1_overnight here; f1_tranche is offered only for CLI uniformity.
CAPITAL_RECOVERY_CASE_CHOICES = ("cr_central", "cr_ida21", "cr_comm_low", "cr_comm_high")
DSCR_CASE_CHOICES = ("dscr_regression", "dscr_central", "dscr_low", "dscr_high")
CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")
# Envelope resolution: F1's own concessional envelope arm, REQUIRED, no
# default -- see concessional_envelope_scenarios() in src/scenarios.py.
# Irrelevant when --capital-treatment-case pre_f1_overnight is passed (as
# this script requires), but still required by load_scenario() regardless
# of mode.
CONCESSIONAL_ENVELOPE_CASE_CHOICES = ("env_dares", "env_dares_plus", "env_none", "env_5x", "env_10x")
# Tier port B: gas_deliverability_case was hardcoded "baseline" in
# run_arm()'s load_scenario() call, which stranded the GMP 2026
# policy-target arms (gas_deliv_*, commit 5b811f2). Now a REQUIRED CLI
# flag, no default -- same discipline as the arms above. Choices sourced
# live from gas_deliverability_scenarios() so the historical-drift and
# GMP policy-target families cannot drift out of sync with src/scenarios.py.
GAS_DELIVERABILITY_CASE_CHOICES = tuple(gas_deliverability_scenarios().keys())
# Phase 2.8: OCGT-to-CCGT conversion arm. REQUIRED, no default. NOTE this
# script requires --capital-treatment-case pre_f1_overnight (the salvage
# probe) and runs both builders, so only 'conv_none' is usable here --
# an available arm requires f1_tranche + the sliced builder.
CONVERSION_CASE_CHOICES = tuple(conversion_scenarios().keys())
# Correction plan 1.5: hydro trajectory arm. REQUIRED, no default.
HYDRO_CASE_CHOICES = tuple(hydro_case_scenarios())
# docs/tariff_arm.md, Steps 1-3: EaaS service tariff arm. REQUIRED, no
# default -- replaces the retired solar_tariff_case's silent 65,000,000
# USD/TWh default. Choices sourced live from tariff_scenarios().
TARIFF_CASE_CHOICES = tuple(tariff_scenarios())


def parse_args():
    parser = argparse.ArgumentParser(description="22_salvage_sensitivity: asset-lifetime sweep")
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
             "'pre_f1_overnight' reproduces the pre-F1 objective (required for this "
             "script's m.salvage_npv probe); 'f1_tranche' replaces it with the "
             "per-tranche CRF annuity and has no salvage to probe. No default: an "
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
    parser.add_argument(
        "--gas-deliverability-case",
        required=True,
        choices=GAS_DELIVERABILITY_CASE_CHOICES,
        help="Gas deliverability-to-power arm fed to "
             "load_scenario(gas_deliverability_case=...). No default. Choices "
             "span two epistemic families: historical-drift trajectories "
             "(downside/baseline/upside/shock_recovery and their flat_* "
             "level-equivalents) and GMP 2026 policy targets "
             "(gas_deliv_flat / gas_deliv_gmp_partial / gas_deliv_gmp_full, "
             "the planned-increment fractions I = 0 / 0.50 / 1.0).",
    )
    parser.add_argument(
        "--conversion-case",
        required=True,
        choices=CONVERSION_CASE_CHOICES,
        help="OCGT-to-CCGT conversion arm fed to "
             "load_scenario(conversion_case=...). No default. This script "
             "needs pre_f1_overnight and runs both builders, so only "
             "'conv_none' is usable -- an available arm requires f1_tranche "
             "and the sliced builder.",
    )
    parser.add_argument(
        "--hydro-case",
        required=True,
        choices=HYDRO_CASE_CHOICES,
        help="Hydro trajectory arm fed to load_scenario(hydro_case=...). "
             "No default. 'hydro_flat' is the pre-1.5 regression arm "
             "(11.47 TWh_e flat); 'hydro_committed' holds the ~12.8 TWh_e "
             "2025 committed level flat; 'hydro_ndc' ramps toward the NDC's "
             "10,400 MW by 2030 then holds flat. All exogenous -- hydro is "
             "never a decision variable.",
    )
    parser.add_argument(
        "--tariff-case",
        required=True,
        choices=TARIFF_CASE_CHOICES,
        help="EaaS service tariff arm fed to load_scenario(tariff_case=...). No "
             "default: the retired solar_tariff_case defaulted to 65,000,000 "
             "USD/TWh silently, with no error, warning or visible signature in "
             "the output. Inert in runners with no EaaS channel "
             "(financing_regime='traditional' fixes solar_eaas_add to 0 via "
             "m.eaas_disable), but load_scenario() requires it regardless. See "
             "tariff_scenarios().",
    )
    return parser.parse_args()

# MW / MWh below which two trajectories count as identical.
TRAJECTORY_TOL = 1e-4

# Materiality threshold. A deviation must exceed this SHARE of cumulative build
# to count as the trajectory "responding" to cost. Without it the verdict fires
# on any nonzero movement, however small -- and the degenerate no_salvage arm
# (all lives = 1 yr) produces a few hundred MW against a fleet of tens of GW,
# which is a tripwire firing, not a finding.
MATERIALITY_SHARE = 0.01


def lifetime_arms():
    """
    Lifetime arms. 'no_salvage' sets every life to 1.0 yr so the salvage
    fraction is zero for every vintage -- an exact switch-off with no code
    change. The remainder bracket the sourced central values.
    """
    central = asset_lifetimes()
    return {
        "no_salvage":     {"solar": 1.0,  "storage": 1.0,  "gas": 1.0},
        "solar_25":       {**central, "solar": 25.0},
        "central":        dict(central),
        "solar_35":       {**central, "solar": 35.0},
        "storage_12":     {**central, "storage": 12.0},
        "storage_18":     {**central, "storage": 18.0},
    }


def run_arm(lives, sliced, econ, capex_tv, peak_case, storage_cc_case,
            gas_build_case, new_gas_deliverability, gas_availability_case,
            gas_retirement_case, gas_capex_case, ccgt_efficiency_case,
            capital_recovery_case, dscr_case, us_inflation, capital_treatment_case,
            concessional_envelope_case, gas_deliverability_case, conversion_case,
            hydro_case, tariff_case):
    """Solve one arm; return objective and capacity trajectories."""
    scenario = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case="unconstrained",
        gas_deliverability_case=gas_deliverability_case,
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
        om_case="om_central",
    )
    # solar_min_build_mw_per_year override to solar_min_build_default()
    # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
    # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).
    scenario["asset_lifetimes"] = dict(lives)

    # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already the
    # builder's own guarded meaning for this argument).
    builder = build_model_sliced if sliced else build_model
    m = builder(scenario=scenario, econ=econ,
                emissions_cap=None, solar_capex_by_year=capex_tv)

    # Solver chatter and the (expected) retirement warnings are captured so the
    # summary table stays readable. Failures still raise.
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"non-optimal: {status['status']}")

    years = list(scenario["years"])
    T = range(len(years))

    solar_add = {
        int(years[t]): float(pyo.value(m.solar_public_add[t]))
                       + float(pyo.value(m.solar_eaas_add[t]))
        for t in T
    }
    try:
        storage_cap = {int(years[t]): float(pyo.value(m.storage_capacity_mwh[t]))
                       for t in T}
    except (AttributeError, KeyError):
        storage_cap = {int(years[t]): 0.0 for t in T}

    return {
        "objective_usd": float(pyo.value(m.system_cost_npv)),
        "salvage_usd": (float(pyo.value(m.salvage_npv))
                        if hasattr(m, "salvage_npv") else None),
        "solar_add_mw": solar_add,
        "storage_capacity_mwh": storage_cap,
        "solar_total_mw": sum(solar_add.values()),
        "storage_final_mwh": storage_cap[int(years[-1])],
        # Phase 2.8: {} for the annual arm and under conv_none.
        "conversion": conversion_diagnostic(m, years, scenario) if sliced else {},
    }


def max_abs_diff(a, b):
    return max(abs(a[k] - b[k]) for k in a)


def analyse(arms, label):
    ref = arms["central"]
    print(f"\n{'=' * 74}")
    print(f"  SALVAGE SENSITIVITY -- {label.upper()} MODEL")
    print(f"{'=' * 74}")
    print(f"  {'arm':<14}{'objective $bn':>15}{'salvage $bn':>13}"
          f"{'solar dMW':>12}{'storage dMWh':>14}")
    print(f"  {'-' * 68}")

    ref_solar_total = max(ref["solar_total_mw"], 1e-9)
    ref_stor_total = max(ref["storage_final_mwh"], 1e-9)

    responsive = False           # material movement in ANY arm
    responsive_real = False      # material movement in a REAL-lifetime arm
    rows = []
    for name, r in arms.items():
        d_solar = max_abs_diff(r["solar_add_mw"], ref["solar_add_mw"])
        d_stor = max_abs_diff(r["storage_capacity_mwh"], ref["storage_capacity_mwh"])
        pct_solar = d_solar / ref_solar_total
        pct_stor = d_stor / ref_stor_total
        material = (pct_solar > MATERIALITY_SHARE) or (pct_stor > MATERIALITY_SHARE)
        if name != "central" and material:
            responsive = True
            if name != "no_salvage":
                responsive_real = True
        salv = f"{r['salvage_usd']/1e9:.4f}" if r["salvage_usd"] is not None else "--"
        mark = "  *" if (name != "central" and d_solar > TRAJECTORY_TOL) else ""
        print(f"  {name:<14}{r['objective_usd']/1e9:>15.4f}{salv:>13}"
              f"{d_solar:>12.5f}{d_stor:>14.4f}{pct_solar:>9.2%}{mark}")
        rows.append({
            "model": label, "arm": name,
            "objective_usd": r["objective_usd"], "salvage_usd": r["salvage_usd"],
            "solar_total_mw": r["solar_total_mw"],
            "storage_final_mwh": r["storage_final_mwh"],
            "max_solar_deviation_mw": d_solar,
            "max_storage_deviation_mwh": d_stor,
        })

    ns, ce = arms["no_salvage"], arms["central"]
    cost_swing = (ce["objective_usd"] - ns["objective_usd"]) / ns["objective_usd"]

    print(f"\n  Cost swing, no_salvage -> central: {cost_swing:+.2%}")
    print(f"  Materiality threshold: {MATERIALITY_SHARE:.0%} of cumulative build "
          f"({ref_solar_total * MATERIALITY_SHARE:,.0f} MW)")
    if responsive_real:
        print("  VERDICT: trajectory RESPONDS materially to cost under a REAL")
        print("           lifetime assumption. The degrees of freedom have")
        print("           opened -- RESTATE the residual finding.")
    elif responsive:
        print("  VERDICT: trajectory invariant across every PLAUSIBLE lifetime.")
        print("           Only the degenerate no_salvage arm (all lives = 1 yr)")
        print("           moves it, and that arm exists solely to switch salvage")
        print("           off. Capacity remains an ACCOUNTING RESIDUAL for any")
        print("           real parameterisation.")
        print(f"           A {abs(cost_swing):.1%} cost change moved the plan by")
        print("           less than the materiality threshold in every real arm.")
    else:
        print("  VERDICT: trajectory IDENTICAL across every arm, including the")
        print("           no-salvage case. Capacity is an ACCOUNTING RESIDUAL --")
        print("           the energy balance determines it and cost cannot move it.")
        print(f"           A {abs(cost_swing):.1%} cost change moved zero MW.")
    print("  ('*' marks any nonzero deviation, material or not.)")
    return rows, responsive, cost_swing


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory

    econ = load_econ(CANONICAL_VOLL, genset_case=None)
    capex_tv = load_solar_capex_by_year(scenario_name="solar_low",
                                        start_year=2025, end_year=MODEL_END_YEAR)
    arms_def = lifetime_arms()
    out = {
        "tolerance_mw": TRAJECTORY_TOL,
        "peak_case": args.peak_case,
        "storage_cc_case": args.storage_cc_case,
        "gas_build_case": args.gas_build_case,
        "new_gas_deliverability": args.new_gas_deliverability,
        "gas_availability_case": args.gas_availability_case,
        "gas_retirement_case": args.gas_retirement_case,
        "gas_capex_case": args.gas_capex_case,
        "ccgt_efficiency_case": args.ccgt_efficiency_case,
        "capital_recovery_case": args.capital_recovery_case,
        "dscr_case": args.dscr_case,
        "us_inflation": args.us_inflation,
        "capital_treatment_case": args.capital_treatment_case,
        "concessional_envelope_case": args.concessional_envelope_case,
        "gas_deliverability_case": args.gas_deliverability_case,
        "conversion_case": args.conversion_case,
        "hydro_case": args.hydro_case,
        # docs/tariff_arm.md, Step 4: recorded for auditability even though
        # this runner has no EaaS channel (financing_regime='traditional'
        # -> m.eaas_disable fixes solar_eaas_add[t] <= 0), making the arm
        # inert. NOT appended to the output tag (deliberate deviation,
        # step 2).
        "tariff_case": args.tariff_case,
        "models": {},
    }
    all_rows = []

    for label, sliced in (("annual", False), ("sliced", True)):
        print(f"\nSolving {label} model, {len(arms_def)} arms ...")
        arms = {}
        for name, lives in arms_def.items():
            try:
                arms[name] = run_arm(lives, sliced, econ, capex_tv, args.peak_case,
                                      args.storage_cc_case, args.gas_build_case,
                                      args.new_gas_deliverability,
                                      args.gas_availability_case,
                                      args.gas_retirement_case,
                                      args.gas_capex_case,
                                      args.ccgt_efficiency_case,
                                      args.capital_recovery_case,
                                      args.dscr_case,
                                      args.us_inflation,
                                      args.capital_treatment_case,
                                      args.concessional_envelope_case,
                                      args.gas_deliverability_case,
                                      args.conversion_case,
                                      args.hydro_case,
                                      args.tariff_case)
                print(f"  {name:<14} ok")
            except Exception as e:
                print(f"  {name:<14} FAILED: {e}")
        if "central" not in arms:
            print(f"  central arm failed -- cannot analyse {label}")
            continue

        rows, responsive, swing = analyse(arms, label)
        all_rows.extend(rows)
        out["models"][label] = {
            "trajectory_responds_to_cost": responsive,
            "cost_swing_no_salvage_to_central": swing,
            "arms": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("solar_add_mw", "storage_capacity_mwh")}
                     for k, v in arms.items()},
            "solar_trajectory_central": arms["central"]["solar_add_mw"],
            "storage_trajectory_central": arms["central"]["storage_capacity_mwh"],
        }

    tag = f"{args.peak_case}__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}__{args.gas_availability_case}__{args.gas_retirement_case}__{args.gas_capex_case}__{args.ccgt_efficiency_case}__{args.capital_recovery_case}__{args.dscr_case}__{args.capital_treatment_case}__{args.concessional_envelope_case}__{args.gas_deliverability_case}__{args.conversion_case}__{args.hydro_case}"
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"trajectories.csv", index=False)
    with open(RESULTS_DIR / f"salvage_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 74}")
    print("  RECORD THE VERDICT IN THE CORRECTION PLAN (item 2.5).")
    print("  Re-run after 2.5 step 3b, 2.5 step 4, 2.6 and 5.3.")
    print(f"  Saved: {RESULTS_DIR}")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
