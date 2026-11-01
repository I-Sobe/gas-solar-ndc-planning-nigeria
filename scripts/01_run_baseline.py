import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.scenarios import (
    load_scenario, asset_lifetime_sweep, gas_deliverability_scenarios,
    conversion_scenarios, hydro_case_scenarios, tariff_scenarios, om_scenarios,
    MODEL_END_YEAR,
)
import pyomo.environ as pyo
from src.optimize_model import solve_model, leakage_diagnostic
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics
from src.io import load_econ, load_solar_capex_by_year
from src.utils import json_safe
from src.output_naming import hash8_of, append_index
from src.cli_arms import complete_arm_tag
from src.summary_years import labelled_decision_totals, labelled_period_totals

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

# ============================================================
# PATH SETUP
# ============================================================

RESULTS_DIR = ROOT / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Plan 0.4, item C (docs/numerical_hygiene_0_4.md): the 1e18 sentinel
# formerly here meant "no cap" only by convention -- build_model() already
# treats emissions_cap=None as "no cap" (the constraint is guarded on it).
# None is the honest value: no emissions_constraint is built at all here,
# rather than a constraint against an astronomical, never-binding number.
EMISSIONS_CAP = None

# 2.6, F0 Step 2.4 (phase_2_4_wiring): peak-demand shape arm is a REQUIRED
# CLI argument, no default -- mirrors the genset_case discipline. An
# omitted arm must fail loudly, not silently build a firm-capacity
# constraint against an unstated demand shape.
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
CAPITAL_RECOVERY_CASE_CHOICES = ("cr_central", "cr_ida21", "cr_comm_low", "cr_comm_high")
DSCR_CASE_CHOICES = ("dscr_regression", "dscr_central", "dscr_low", "dscr_high")
CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")
# Envelope resolution: F1's own concessional envelope arm, REQUIRED, no
# default -- see concessional_envelope_scenarios() in src/scenarios.py.
CONCESSIONAL_ENVELOPE_CASE_CHOICES = ("env_dares", "env_dares_plus", "env_none", "env_5x", "env_10x")
# Tier port B: gas_deliverability_case was hardcoded "baseline" in the
# load_scenario() call below, which stranded the GMP 2026 policy-target
# arms (gas_deliv_*, commit 5b811f2). Now a REQUIRED CLI flag, no
# default -- same discipline as the arms above. Choices sourced live
# from gas_deliverability_scenarios() so the two epistemic families
# (historical drift + GMP policy targets) cannot drift out of sync with
# src/scenarios.py.
GAS_DELIVERABILITY_CASE_CHOICES = tuple(gas_deliverability_scenarios().keys())
# Phase 2.8: OCGT-to-CCGT conversion arm. REQUIRED, no default. NOTE this
# is the ANNUAL baseline runner -- conversion is slice-resolved and
# build_model() raises on any available arm, so only 'conv_none' will
# solve here; use 01b_run_baseline_sliced.py for conversion runs.
CONVERSION_CASE_CHOICES = tuple(conversion_scenarios().keys())
# Correction plan 1.5: hydro trajectory arm. REQUIRED, no default.
HYDRO_CASE_CHOICES = tuple(hydro_case_scenarios())
# docs/tariff_arm.md, Steps 1-3: EaaS service tariff arm. REQUIRED, no
# default -- replaces the retired solar_tariff_case's silent 65,000,000
# USD/TWh default. Choices sourced live from tariff_scenarios().
TARIFF_CASE_CHOICES = tuple(tariff_scenarios())
# task_om_implementation: fixed + variable O&M arm. REQUIRED, no default --
# replaces the retired STORAGE_OM_PER_MWH_YR, which never reached the
# objective for any technology. Choices sourced live from om_scenarios().
OM_CASE_CHOICES = tuple(om_scenarios())


def parse_args():
    parser = argparse.ArgumentParser(description="01_run_baseline: unconstrained baseline solve")
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
             "load_scenario(conversion_case=...). No default. 'conv_none' is "
             "the pre-2.8 regression arm. Conversion dispatch is "
             "slice-resolved and lives in build_model_sliced() only -- this "
             "ANNUAL runner raises on any available arm; use "
             "01b_run_baseline_sliced.py for conversion runs.",
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
    parser.add_argument(
        "--om-case",
        required=True,
        choices=OM_CASE_CHOICES,
        help="Fixed + variable O&M arm fed to load_scenario(om_case=...). No "
             "default: the retired STORAGE_OM_PER_MWH_YR was defined in io.py "
             "but never read by either builder, for any technology, with no "
             "error or visible signature in the output. See om_scenarios().",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()
    tag = f"{args.peak_case}__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}__{args.gas_availability_case}__{args.gas_retirement_case}__{args.gas_capex_case}__{args.ccgt_efficiency_case}__{args.capital_recovery_case}__{args.dscr_case}__{args.capital_treatment_case}__{args.concessional_envelope_case}__{args.gas_deliverability_case}__{args.conversion_case}__{args.hydro_case}"
    # TASK 53 2.5 (G7-4): append every arm the tag omits (om_case, reserve_margin, tariff_case, us_inflation, window, storage).
    tag = complete_arm_tag(tag, args, in_tag={"peak_multiple_case","storage_cc_case","gas_build_case","new_gas_deliverability","gas_availability_case","gas_retirement_case","gas_capex_case","ccgt_efficiency_case","capital_recovery_case","dscr_case","capital_treatment_case","concessional_envelope_case","gas_deliverability_case","conversion_case","hydro_case"})
    # TASK 53 2.5: the completed tag exceeds the ~255-byte filename limit (412 bytes with the "diagnostics__" prefix and ".json");
    # files carry hash8 instead, the FULL tag is written into the JSON ("arm_tag") and recorded in INDEX.md (as 01b does).
    hash8 = hash8_of(tag)
    append_index(RESULTS_DIR, hash8, tag)

    scenario = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case="unconstrained",
        gas_deliverability_case=args.gas_deliverability_case,
        solar_build_case="deployment_unconstrained",
        land_case="loose",
        carbon_case="no_policy",
        start_year=2025,
        end_year=MODEL_END_YEAR,
        peak_multiple_case=args.peak_case,
        reserve_margin="nirp_spinning",
        storage_cc_case=args.storage_cc_case,
        gas_build_case=args.gas_build_case,
        new_gas_deliverability=args.new_gas_deliverability,
        gas_availability_case=args.gas_availability_case,
        gas_retirement_case=args.gas_retirement_case,
        gas_capex_case=args.gas_capex_case,
        ccgt_efficiency_case=args.ccgt_efficiency_case,
        capital_recovery_case=args.capital_recovery_case,
        dscr_case=args.dscr_case,
        us_inflation=args.us_inflation,
        capital_treatment_case=args.capital_treatment_case,
        concessional_envelope_case=args.concessional_envelope_case,
        conversion_case=args.conversion_case,
        hydro_case=args.hydro_case,
        tariff_case=args.tariff_case,
        om_case=args.om_case,
    )

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    years = scenario["years"]

    # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
    # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    # solar_min_build_mw_per_year override to solar_min_build_default()
    # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
    # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).

    # Build and solve the baseline model with no emissions cap at all
    # (EMISSIONS_CAP = None; no m.emissions_constraint is built).
    m = build_model_sliced(
        scenario=scenario,
        econ=econ,
        emissions_cap=EMISSIONS_CAP,
        emissions_cap_by_year=None,
        solar_capex_by_year=solar_capex_tv,
    )

    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"Baseline solve failed: {status}")

    diag = extract_planning_diagnostics(m, scenario, econ)

    # Decision variables
    dv = {
        "solar_add_mw_by_year":
            {int(y):
                float(pyo.value(m.solar_public_add[t]))
                + float(pyo.value(m.solar_eaas_add[t]))
            for t,y in enumerate(years)
        },
        # gas_add removed (Fix 9): gas generation is fuel-constrained,
        # new gas capacity additions are never optimal and the variable
        # was removed from the model.
        "storage_add_mwh_by_year": {
            int(y): float(pyo.value(m.storage_add[t]))
            for t, y in enumerate(years)
        },
    }
    # TASK 53 2.4 (G1-2): year-labelled aggregates with their 2045 counterparts.
    dv.update(labelled_decision_totals(
        years, dv["solar_add_mw_by_year"],
        {int(y): float(pyo.value(m.storage_capacity_mwh[t])) for t, y in enumerate(years)}))

    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    _period = labelled_period_totals(years, diag["unserved_twh_by_year"], diag["emissions_tco2_by_year"])
    assert abs(_period[f"cumulative_emissions_{int(years[0])}_{int(years[-1])}_tco2"] - float(pyo.value(m.emissions))) <= 1e-6 * max(1.0, float(pyo.value(m.emissions))), \
        "sum of emissions_tco2_by_year != m.emissions"
    # Leakage guard (2.6, D15), always skipped here: this baseline runner
    # carries no emissions commitment (EMISSIONS_CAP = None, no cap built),
    # so a leakage share has nothing to be measured against. Plan 0.4,
    # item C: this line previously special-cased the 1e18 sentinel
    # (`EMISSIONS_CAP > 1e15 and False` -- the trailing `and False` already
    # forced this to always evaluate False, i.e. leak was already
    # unconditionally None here before this task). Simplified to state
    # that directly rather than comparing None against a numeric threshold.
    leak = None
    if leak and leak["flagged"]:
        print(f"  LEAKAGE FLAG: {leak['leakage_share']:.1%} of emissions "
              f"({leak['leakage_tco2']/1e6:.1f} MtCO2) fall outside the cap "
              f"boundary. NOT an NDC-compliant plan.")

    print("Storage discharge (TWh by year):", diag["storage_discharge_twh_e_by_year"])
    print("Storage binding constraint by year:", diag["storage_binding_by_year"])

    # ------------------------------------------------------------
    # Save diagnostics (json)
    # ------------------------------------------------------------
    with open(RESULTS_DIR / f"diagnostics__{hash8}.json", "w") as f:
        json.dump(json_safe({**diag, "arm_tag": tag}), f, indent=2)

    # Compute actual discounted public solar CAPEX used.
    # The model uses econ["SOLAR_CAPEX_PER_MW"] — a fixed 2025 scalar from
    # the solar_low NREL ATB scenario. There is no time-varying solar_capex_param
    # Pyomo object in the current model; that was a planned extension that was
    # not implemented. Using the scalar is consistent with how the objective
    # function prices solar additions.
    solar_public_npv_spend = sum(
        float(pyo.value(m.DF[t]))
        * float(pyo.value(m.solar_public_add[t]))
        * float(pyo.value(m.solar_capex_param[t])) #time-varying NREL ATB value
        for t in range(len(years))
    )

    print("Discounted public solar CAPEX used (USD):", solar_public_npv_spend)


    # ------------------------------------------------------------
    # Save summary (json)
    # ------------------------------------------------------------
    summary = {
        "cap_scenario": "baseline_no_policy",
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
        # financing_regime='traditional' here makes the arm inert
        # (m.eaas_disable fixes solar_eaas_add[t] <= 0) -- NOT appended to
        # the output tag (deliberate deviation, step 2).
        "tariff_case": args.tariff_case,
        # task_om_implementation: recorded for auditability, like tariff_case
        # above -- NOT appended to the output tag (same deliberate deviation).
        "om_case": args.om_case,
        "decision_variables": dv,
        "npv_total_cost_usd": npv_total_cost_usd,
        "npv_gross_capex_report_window_usd": diag.get("npv_gross_capex_report_window_usd"),
        "npv_gross_capex_full_horizon_usd":  diag.get("npv_gross_capex_full_horizon_usd"),
        "report_window_end_year":            diag.get("report_window_end_year"),
        "model_horizon_end_year":            diag.get("model_horizon_end_year"),
        "npv_total_cost_window": f"{int(years[0])}-{int(years[-1])}, discounted to {int(years[0])}",
        **_period,
        "solar_public_npv_spend": solar_public_npv_spend,
        "leakage": leak,
        "notes": "Baseline = no emissions cap (emissions_cap=None, no emissions_constraint built).",
        "arm_tag": tag,
    }

    with open(RESULTS_DIR / f"summary__{hash8}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------
    # Save yearly timeseries (csv)
    # ------------------------------------------------------------
    years = scenario["years"]

    ts = pd.DataFrame(
        {
            "year": [int(y) for y in years],
            "demand_twh": [diag["demand_twh_by_year"][int(y)] for y in years],
            "gas_avail_twh_th": [diag["gas_avail_twh_th_by_year"][int(y)] for y in years],
            "gas_to_power_twh_th": [diag["fleet_gas_to_power_twh_th_by_year"][int(y)] for y in years],
            "gas_generation_twh_e": [diag["fleet_gas_generation_twh_e_by_year"][int(y)] for y in years],
            "solar_generation_twh_e": [diag["solar_generation_twh_e_by_year"][int(y)] for y in years],
            "storage_discharge_twh_e": [diag["storage_discharge_twh_e_by_year"][int(y)] for y in years],
            "unserved_twh": [diag["unserved_twh_by_year"][int(y)] for y in years],
            "emissions_tco2": [diag["emissions_tco2_by_year"][int(y)] for y in years],
            "gas_shadow_pv_usd_per_twh_th": [diag["gas_shadow_pv_usd_per_twh_th_by_year"][int(y)] for y in years],
            "gas_shadow_yr_usd_per_twh_th": [diag["gas_shadow_yr_usd_per_twh_th_by_year"][int(y)] for y in years],
            "carbon_shadow_pv_usd_per_tco2": [diag["carbon_shadow_pv_usd_per_tco2_by_year"][int(y)] for y in years],
            "carbon_shadow_yr_usd_per_tco2": [diag["carbon_shadow_yr_usd_per_tco2_by_year"][int(y)] for y in years],
            "discount_factor": [diag["discount_factor_by_year"][int(y)] for y in years],
            # gas_add_mw removed (Fix 9): variable no longer exists in model
            "solar_capacity_mw": [float(pyo.value(m.solar_capacity_mw[t])) for t in range(len(years))],
            "gas_capacity_mw": [float(pyo.value(m.gas_capacity_mw[t])) for t in range(len(years))],
            "storage_capacity_mwh": [float(pyo.value(m.storage_capacity_mwh[t])) for t in range(len(years))],
            "subsidy_per_mw_usd": [diag["subsidy_per_mw_usd_by_year"][y] for y in years],
            "selfgen_twh": [diag["selfgen_twh_by_year"][int(y)] for y in years],        }
    )

    ts.to_csv(RESULTS_DIR / f"timeseries__{hash8}.csv", index=False)

    print("--- Baseline run saved ---")
    print("Saved diagnostics:", RESULTS_DIR / f"diagnostics__{hash8}.json")
    print("Saved summary:", RESULTS_DIR / f"summary__{hash8}.json")
    print("Saved timeseries:", RESULTS_DIR / f"timeseries__{hash8}.csv")
    print("Solar addition (MW/year):", dv["solar_add_mw_by_year"])
    print(f"Storage capacity {int(years[-1])} (MWh):", dv[f"storage_capacity_{int(years[-1])}_mwh"])

if __name__ == "__main__":
    main()
