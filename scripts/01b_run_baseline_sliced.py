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
from src.optimize_model_sliced import (
    build_model_sliced, annual_totals, slice_detail, SLICES,
    firm_capacity_diagnostic, conversion_diagnostic,
)
from src.io import load_econ, load_solar_capex_by_year
from src.output_naming import hash8_of, append_index, om_tag_suffix
from src.cli_arms import complete_arm_tag
from src.utils import json_safe
from src.summary_years import labelled_decision_totals, labelled_period_totals
# task_capex_reporting_field, Part 2: extract_planning_diagnostics() is
# NOT annual-only (the in-file comment below this import list said so;
# task_capex_reporting_field Part 1A found that false by dimensionality
# audit, and 18_run_pareto_frontier.py / 28_run_capital_frontier.py are
# already-committed, working precedent calling it on sliced,
# conversion-active models). Imported ONLY to reuse its already-correct
# npv_gross_capex_report_window_usd / npv_gross_capex_full_horizon_usd
# fields (built from _capex_at(), which this runner cannot import
# directly -- it is a closure local to extract_planning_diagnostics(),
# not a module-level name). Every other field this module computes
# inline stays inline; this is not a switch to calling the shared
# function for everything.
from src.optimize_experiments import extract_planning_diagnostics

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

RESULTS_DIR = ROOT / "results" / "baseline_sliced"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Plan 0.4, item C (docs/numerical_hygiene_0_4.md): the 1e18 sentinel
# formerly here meant "no cap" only by convention -- build_model_sliced()
# already treats emissions_cap=None as "no cap" (the constraint is guarded
# on it). None is the honest value: no emissions_constraint is built at
# all here, rather than a constraint against an astronomical, never-
# binding number.
EMISSIONS_CAP = None

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
# Phase 2.8: OCGT-to-CCGT conversion arm. REQUIRED, no default -- same
# discipline as the arms above. conv_none is the pre-2.8 regression arm.
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
    parser = argparse.ArgumentParser(description="01b_run_baseline_sliced: unconstrained sliced baseline solve")
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
             "the pre-2.8 regression arm (conversion unavailable); "
             "'conv_central' / 'conv_low' / 'conv_high' set retrofit capex "
             "(USD/kW nameplate), capacity uplift and retrofit efficiency. "
             "An available arm requires --capital-treatment-case f1_tranche.",
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
    parser.add_argument(
        "--social-discount-rate",
        type=float,
        default=None,
        help="Override of scenario['social_discount_rate'] (the objective's discount rate r_s; the scenario "
             "default is 0.08). OPTIONAL, and it enters the output key whenever it is GIVEN, including at "
             "0.08, so the discount-rate sweep can never overwrite the central anchor (a G7-4 instance: a "
             "parameter that changes the model must reach the key). Omitted: the key and every stored hash "
             "are unchanged. scripts/run_chapter3_reruns.sh cell 3 is its only caller.",
    )
    parser.add_argument(
        "--gas-vintage-dispatch",
        choices=("on", "off"),
        default="on",
        help="TASK 74: ON by default (register L123). Capacity added in the model converts at "
             "gas_new_efficiency (the ccgt_efficiency_case value) and pre-existing capacity at gas_eta; "
             "emissions follow the thermal draw. Effective only at new_gas_deliverability=constrained (forced "
             "off under exempt). 'off' reproduces the pre-TASK-74 blended dispatch. The EFFECTIVE value enters "
             "the output key either way. With --gas-new-efficiency-override, results go to "
             "results/vintage_dispatch_sensitivity/, never to results/baseline_sliced/.",
    )
    parser.add_argument(
        "--gas-new-efficiency-override",
        type=float,
        default=None,
        help="With --gas-vintage-dispatch only: replace gas_new_efficiency by this value (0.287 reproduces the "
             "blended-dispatch model exactly, which is the regression check of TASK 65 B.2). Enters the key.",
    )
    parser.add_argument(
        "--conc-share-rule",
        choices=("split_prorata", "split_frontload", "split_zero", "split_central_realised"),
        default=None,
        help="TASK 98: SUPPLY the concessional-commercial division instead of letting the model determine it (Chapter 1 "
             "Question 2's counterfactual). Optional, default inert: absent, the key and every stored hash are unchanged. "
             "split_prorata = constant share exhausting the envelope at the reference plan; split_frontload = whole envelope in "
             "the first model year; split_zero = no concessional capital; split_central_realised = 1.0 at --conc-share-realised-year. "
             "See src/supplied_division.py. Results go to results/supplied_division/, never to results/baseline_sliced/.",
    )
    parser.add_argument("--conc-share-reference-summary", default=None,
                        help="split_prorata only: summary__<hash8>.json of the endogenous partner whose solar and storage additions calibrate the share.")
    parser.add_argument("--conc-share-realised-year", type=int, default=2035,
                        help="split_central_realised only: the single vintage supplied at share 1.0 (default 2035, the central's own).")
    parser.add_argument("--conc-share-first-year", type=int, default=None,
                        help="split_frontload only: the first eligible vintage (default: the first model year).")
    parser.add_argument(
        "--hydro-om-term",
        action="store_true",
        help="TASK 69 B.1 (register L95), off by default: build a hydro fixed O&M term in the objective. The rate is "
             "--hydro-fom-usd-per-kw-yr, whose only value set anywhere is the null 0.0 (no sourced rate exists yet). "
             "Results go to results/deferred_flag_sensitivity/, never to results/baseline_sliced/, and the flag enters the key.",
    )
    parser.add_argument(
        "--hydro-fom-usd-per-kw-yr",
        type=float,
        default=0.0,
        help="With --hydro-om-term only: hydro fixed O&M in USD per kW-yr. Default 0.0, the null value that must "
             "reproduce the anchor exactly. Do not set a non-zero value until the parameter has a provenance row.",
    )
    parser.add_argument(
        "--selfgen-busbar-factor",
        type=float,
        default=None,
        help="TASK 69 B.2 (register L96): override of scenario['selfgen_busbar_factor'] (default 1.0, the value every "
             "stored result used; 1/(1-0.07) = 1.0753 would credit a 7%% transmission loss). Omitted: nothing changes. "
             "Given (including 1.0): results go to results/deferred_flag_sensitivity/ and the value enters the key.",
    )
    parser.add_argument("--print-hash8", action="store_true",
                        help="print the output hash8 for these arms and exit (no build, no solve)")
    return parser.parse_args()


def main():
    global RESULTS_DIR

    args = parse_args()
    gvd = args.gas_vintage_dispatch == "on" and args.new_gas_deliverability == "constrained"
    if args.gas_new_efficiency_override is not None and not gvd:
        raise SystemExit("--gas-new-efficiency-override requires gas vintage dispatch on (and constrained deliverability)")
    if args.hydro_om_term is False and args.hydro_fom_usd_per_kw_yr != 0.0:
        raise SystemExit("--hydro-fom-usd-per-kw-yr requires --hydro-om-term")
    if args.conc_share_rule == "split_prorata" and not args.conc_share_reference_summary:
        raise SystemExit("--conc-share-rule split_prorata requires --conc-share-reference-summary")
    if args.conc_share_rule is None and (args.conc_share_reference_summary or args.conc_share_first_year is not None):
        raise SystemExit("--conc-share-reference-summary / --conc-share-first-year require --conc-share-rule")
    if args.conc_share_rule is not None:
        RESULTS_DIR = ROOT / "results" / "supplied_division"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    elif args.gas_new_efficiency_override is not None:
        RESULTS_DIR = ROOT / "results" / "vintage_dispatch_sensitivity"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    elif args.hydro_om_term or args.selfgen_busbar_factor is not None:
        RESULTS_DIR = ROOT / "results" / "deferred_flag_sensitivity"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.peak_case}__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}__{args.gas_availability_case}__{args.gas_retirement_case}__{args.gas_capex_case}__{args.ccgt_efficiency_case}__{args.capital_recovery_case}__{args.dscr_case}__{args.capital_treatment_case}__{args.concessional_envelope_case}__{args.gas_deliverability_case}__{args.conversion_case}__{args.hydro_case}" + om_tag_suffix(args.om_case)
    # TASK 53 2.5 (G7-4): the arms the tag above omits (reserve_margin, tariff_case, us_inflation, capital_window_case,
    # storage_capex_case) are appended, so no arm can change without changing the output key. This moves the hash.
    if args.social_discount_rate is not None:
        tag += f"__social_discount_rate={args.social_discount_rate}"
    if args.gas_new_efficiency_override is not None:
        tag += f"__gas_new_efficiency_override={args.gas_new_efficiency_override}"
    if args.conc_share_rule is not None:    # TASK 98 (G7-4): the supplied division reaches the output key, with the data that calibrates it
        tag += f"__conc_share_rule={args.conc_share_rule}"
        if args.conc_share_rule == "split_prorata":
            tag += f"__conc_share_reference={Path(args.conc_share_reference_summary).stem}"
        if args.conc_share_rule == "split_central_realised":
            tag += f"__conc_share_realised_year={args.conc_share_realised_year}"
        if args.conc_share_rule == "split_frontload" and args.conc_share_first_year is not None:
            tag += f"__conc_share_first_year={args.conc_share_first_year}"
    if args.hydro_om_term:
        tag += f"__hydro_om_term=on__hydro_fom_usd_per_kw_yr={args.hydro_fom_usd_per_kw_yr}"
    if args.selfgen_busbar_factor is not None:
        tag += f"__selfgen_busbar_factor={args.selfgen_busbar_factor}"
    tag = complete_arm_tag(tag, args, in_tag={"peak_multiple_case","storage_cc_case","gas_build_case","new_gas_deliverability","gas_availability_case","gas_retirement_case","gas_capex_case","ccgt_efficiency_case","capital_recovery_case","dscr_case","capital_treatment_case","concessional_envelope_case","gas_deliverability_case","conversion_case","hydro_case","om_case"}, extra={"gas_vintage_dispatch": gvd})
    # docs/output_naming.md: the full tag above exceeds the filesystem's
    # ~255-byte per-path-component limit once enough required arms
    # accumulate (it did, here -- OSError [Errno 36] on the diagnostics
    # write). Path components use a short deterministic hash8 instead;
    # the FULL tag is preserved as "arm_tag" in every JSON and as a
    # header comment in every CSV (see each write site below), and
    # recorded in RESULTS_DIR/INDEX.md so the directory stays navigable.
    hash8 = hash8_of(tag)
    if args.print_hash8:
        # TASK 53: lets a driver derive the output hash instead of pinning a constant that every
        # tag change silently invalidates (run_phase_b.sh's ITEM0_CENTRAL_DIAG_HASH8 was one).
        print(hash8)
        return
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
        gas_vintage_dispatch=gvd,
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

    if args.social_discount_rate is not None:
        assert float(scenario["social_discount_rate"]) == 0.08, scenario["social_discount_rate"]
        scenario["social_discount_rate"] = float(args.social_discount_rate)
        print(f"SDR OVERRIDE APPLIED: social_discount_rate 0.08 -> {args.social_discount_rate}", flush=True)

    if args.selfgen_busbar_factor is not None:
        scenario["selfgen_busbar_factor"] = float(args.selfgen_busbar_factor)
        print(f"SELFGEN BUSBAR FACTOR OVERRIDE: {scenario['selfgen_busbar_factor']}", flush=True)
    if args.conc_share_rule is not None:
        _rule = {"name": args.conc_share_rule}
        if args.conc_share_rule == "split_prorata":
            _ref = json.loads(Path(args.conc_share_reference_summary).read_text())["decision_variables"]
            _rule["reference"] = {"solar_add_mw_by_year": _ref["solar_add_mw_by_year"], "storage_add_mwh_by_year": _ref["storage_add_mwh_by_year"]}
            _rule["reference_file"] = str(args.conc_share_reference_summary)
        if args.conc_share_rule == "split_central_realised":
            _rule["realised_year"] = args.conc_share_realised_year
        if args.conc_share_rule == "split_frontload" and args.conc_share_first_year is not None:
            _rule["first_eligible_year"] = args.conc_share_first_year
        scenario["conc_share_rule"] = _rule
        print(f"SUPPLIED CONCESSIONAL DIVISION: rule {args.conc_share_rule} (the model does not determine the division)", flush=True)
    if args.hydro_om_term:
        scenario["hydro_om_term"] = True
        scenario["hydro_fom_usd_per_kw_yr"] = float(args.hydro_fom_usd_per_kw_yr)
        print(f"HYDRO O&M TERM ON: hydro FOM {scenario['hydro_fom_usd_per_kw_yr']} USD/kW-yr", flush=True)
    if gvd:
        assert scenario["gas_vintage_dispatch"] is True
        if args.gas_new_efficiency_override is not None:
            scenario["gas_new_efficiency"] = float(args.gas_new_efficiency_override)
        print(f"GAS VINTAGE DISPATCH ON: added capacity converts at {scenario['gas_new_efficiency']}, "
              f"pre-existing at gas_eta {scenario['gas_eta']}", flush=True)

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    years = scenario["years"]

    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    # solar_min_build_mw_per_year override to solar_min_build_default()
    # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
    # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).

    # ------------------------------------------------------------
    # Build and solve the SLICED baseline
    # ------------------------------------------------------------
    m = build_model_sliced(
        scenario=scenario,
        econ=econ,
        emissions_cap=EMISSIONS_CAP,
        emissions_cap_by_year=None,
        solar_capex_by_year=solar_capex_tv,
    )

    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"Sliced baseline solve failed: {status}")

    # ------------------------------------------------------------
    # Aggregate sliced operational variables to annual totals.
    # NOTE: extract_planning_diagnostics is annual-only and would crash on the
    # sliced [t,(s,p)] indexing, so the sliced runner builds its diagnostics
    # directly from annual_totals plus the (unchanged) annual variables.
    # ------------------------------------------------------------
    tot = annual_totals(m, years)  # dict: name -> {year: TWh}

    def annual_series(name):
        return {int(y): tot[name][int(y)] for y in years}

    # Phase 2.8: annual_totals()/"gas_generation" is the UNCONVERTED fleet
    # stream only (eta * gas_to_power). Converted plant is a separate
    # stream (docs/phase_2_8_conversion.md "Known reporting gap"); its
    # per-tier converted_generation_twh_e_by_year is summed here so the
    # backcast -- the calibration instrument -- compares TOTAL gas-side
    # output against observed 2024, not just the fleet stream. {} / 0.0
    # under conv_none.
    _conv_diag = conversion_diagnostic(m, years, scenario)

    def converted_gen(y):
        if not _conv_diag:
            return 0.0
        return sum(
            _conv_diag[tier]["converted_generation_twh_e_by_year"].get(int(y), 0.0)
            for tier in ("A", "B", "C")
        )

    # Duals (annual constraints, unchanged from the annual model).
    # TASK 53 2.3: the dual inherits the objective's present-value units (L101, L106). The
    # solver's number is the _pv_ series (year-0 USD per TWh_th); _yr_ divides by m.DF[t].
    gas_shadow = {}
    gas_shadow_yr = {}
    for t, y in enumerate(years):
        d = m.dual.get(m.gas_balance[t], None)
        gas_shadow[int(y)] = abs(float(d)) if d is not None else None
        gas_shadow_yr[int(y)] = (gas_shadow[int(y)] / float(pyo.value(m.DF[t]))) if d is not None else None

    # Emissions per year (Expression, annual).
    emissions_by_year = {
        int(y): float(pyo.value(m.emissions_by_year[t]))
        for t, y in enumerate(years)
    }

    # ------------------------------------------------------------
    # Decision variables (all annual — unchanged)
    # ------------------------------------------------------------
    dv = {
        "solar_add_mw_by_year": {
            int(y): float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
            for t, y in enumerate(years)
        },
                "gas_add_mw_by_year": {
            int(y): float(pyo.value(m.gas_add[t])) for t, y in enumerate(years)
        },
        "storage_add_mwh_by_year": {
            int(y): float(pyo.value(m.storage_add[t]))
            for t, y in enumerate(years)
        },
    }
    # TASK 53 2.4 (G1-2): every aggregate carries the year it is taken from and every
    # last-model-year value carries its 2045 counterpart. Replaces the unlabelled
    # solar_total_built_mw / gas_total_built_mw / final_storage_capacity_mwh.
    dv.update(labelled_decision_totals(
        years, dv["solar_add_mw_by_year"],
        {int(y): float(pyo.value(m.storage_capacity_mwh[t])) for t, y in enumerate(years)},
        gas_add_by_year=dv["gas_add_mw_by_year"]))

    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    # TASK 53 2.4 (G1-2): period totals named by their window, 2025..end and 2025..2045.
    _period = labelled_period_totals(years, annual_series("unserved"), emissions_by_year)
    _emissions_model_total = float(pyo.value(m.emissions))
    assert abs(_period[f"cumulative_emissions_{int(years[0])}_{int(years[-1])}_tco2"] - _emissions_model_total) <= 1e-6 * max(1.0, abs(_emissions_model_total)), \
        "sum of emissions_by_year != m.emissions"

    # F2 Step 3: concessional-draw fields (envelope resolution, "F2:
    # asset lifetime restored"). This runner does not call
    # extract_planning_diagnostics (annual-only, see the NOTE above), so
    # they were never surfaced here -- add them directly. f1_tranche only;
    # m.concessional_volume_by_vintage and the *_conc_vol/*_comm_vol Var
    # pairs are structurally absent under pre_f1_overnight. Denominator
    # sums only concessional-eligible channels (solar public, solar EaaS,
    # storage -- gas excluded, envelope resolution Decision 1), matching
    # src/optimize_experiments.py's extract_planning_diagnostics.
    if scenario["capital_treatment_case"] == "f1_tranche":
        concessional_volume_by_vintage_usd = {
            int(y): float(pyo.value(m.concessional_volume_by_vintage[t]))
            for t, y in enumerate(years)
        }
        concessional_share_by_vintage = {}
        for t, y in enumerate(years):
            _eligible_vol_t = (
                float(pyo.value(m.solar_public_conc_vol[t])) + float(pyo.value(m.solar_public_comm_vol[t]))
                + float(pyo.value(m.solar_eaas_conc_vol[t])) + float(pyo.value(m.solar_eaas_comm_vol[t]))
                + float(pyo.value(m.storage_conc_vol[t])) + float(pyo.value(m.storage_comm_vol[t]))
            )
            _conc_vol_t = concessional_volume_by_vintage_usd[int(y)]
            concessional_share_by_vintage[int(y)] = (
                _conc_vol_t / _eligible_vol_t if _eligible_vol_t > 0 else 0.0
            )
    else:
        concessional_volume_by_vintage_usd = None
        concessional_share_by_vintage = None
    # Leakage guard (2.6, D15), always skipped here: this sliced baseline
    # runner carries no emissions commitment (EMISSIONS_CAP = None, no cap
    # built), so a leakage share has nothing to be measured against. Plan
    # 0.4, item C: this line previously special-cased the 1e18 sentinel
    # (`EMISSIONS_CAP < 1e15` -- always False when EMISSIONS_CAP was 1e18,
    # i.e. leak was already unconditionally None here before this task).
    # Simplified to state that directly rather than comparing None against
    # a numeric threshold.
    leak = None
    if leak and leak["flagged"]:
        print(f"  LEAKAGE FLAG: {leak['leakage_share']:.1%} of emissions "
              f"({leak['leakage_tco2']/1e6:.1f} MtCO2) fall outside the cap "
              f"boundary. NOT an NDC-compliant plan.")
        
    # ------------------------------------------------------------
    # BACKCAST CHECK — printed prominently
    # ------------------------------------------------------------
    gas_fleet_2025 = annual_series("gas_generation")[2025]
    gas_conv_2025 = converted_gen(2025)
    gas_2025 = gas_fleet_2025 + gas_conv_2025   # TOTAL gas-side (fleet + converted)
    print("=" * 55)
    print("SLICED BASELINE — BACKCAST CHECK (2025 vs observed 2024)")
    print(f"  gas generation : {gas_2025:6.2f} TWh_e   (observed 25.62)"
          f"   (fleet {gas_fleet_2025:.2f} + converted {gas_conv_2025:.2f})")
    print(f"  deviation      : {(gas_2025 - 25.62) / 25.62:+6.1%}")
    print(f"  unserved 2025  : {annual_series('unserved')[2025]:.4f} TWh")
    print(f" selfgen 2025    : {annual_series('selfgen')[2025]:.4f} TWh")
    print(f"  objective      : {npv_total_cost_usd / 1e9:.3f} bn USD")
    print(f"  storage {int(years[-1])}   : {dv[f'storage_capacity_{int(years[-1])}_mwh']:,.0f} MWh")
    print("=" * 55)

    # ------------------------------------------------------------
    # Per-slice detail for the first and terminal model years
    # (peak-adequacy inspection). L15: was hardcoded "2045", stale since
    # the horizon extension -- both years are read dynamically now.
    # ------------------------------------------------------------
    slice_2025 = slice_detail(m, years, 0)
    slice_terminal_year = slice_detail(m, years, len(years) - 1)

    # ------------------------------------------------------------
    # Reporting-window / full-horizon gross capex (task_capex_reporting_
    # field, Part 2). Calls the shared extract_planning_diagnostics() for
    # these two fields ONLY, reusing its _capex_at() (solar + storage +
    # gas + Phase 2.8 conversion, task_conversion_capex_exposure) rather
    # than re-deriving the same arithmetic here a second time. Passing
    # solar_capex_tv (already loaded above, same series the model itself
    # was built with) so this does not silently reload a possibly-
    # different series under scenario.get("solar_capex_scenario",
    # "solar_low")'s default.
    _planning_diag = extract_planning_diagnostics(
        m, scenario, econ, solar_capex_series=solar_capex_tv
    )
    npv_gross_capex_report_window_usd = _planning_diag["npv_gross_capex_report_window_usd"]
    npv_gross_capex_full_horizon_usd = _planning_diag["npv_gross_capex_full_horizon_usd"]

    # TASK 60: the objective's decomposition by term, written to this runner's diagnostics.
    # The block has the term structure of extract_planning_diagnostics()'s `cost_decomposition`
    # (src/optimize_experiments.py), whose own block is carried unchanged under the same keys so the sliced
    # and annual outputs stay comparable; this runner previously called that function for the two gross
    # capex fields only and discarded the rest. `components_unrounded` re-reads the model's m.cost_*_npv
    # Expressions at full precision and `reconciliation` states the remainder against the objective
    # instead of absorbing it. Read-only: nothing here feeds the model, the summary or decision_variables.
    cost_decomposition = dict(_planning_diag["cost_decomposition"])
    if scenario["capital_treatment_case"] == "f1_tranche":
        def _term(name):
            return float(pyo.value(getattr(m, name))) if hasattr(m, name) else 0.0
        _terms = {
            "gas_opex_npv":                    _term("cost_gas_opex_npv"),
            "gas_capital_charge_npv":          _term("cost_gas_capital_charge_npv"),
            "public_solar_capital_charge_npv": _term("cost_public_solar_capital_charge_npv"),
            "eaas_solar_capital_charge_npv":   _term("cost_eaas_solar_capital_charge_npv"),
            "storage_capital_charge_npv":      _term("cost_storage_capital_charge_npv"),
            "rebuild_capital_charge_npv":      _term("cost_rebuild_capital_charge_npv"),
            "conversion_capital_charge_npv":   _term("cost_conversion_capital_charge_npv"),
            "capital_charge_total_full_horizon_npv": _term("capital_charge_total_npv"),
            "selfgen_npv":                     _term("cost_selfgen_npv"),
            "voll_penalty_npv":                _term("cost_unserved_voll_npv"),
            "carbon_cost_npv":                 _term("cost_carbon_npv"),
            "om_cost_npv":                     _term("cost_om_npv"),
            "salvage_npv":                     _term("salvage_npv"),
        }
        # TASK 62 (invariant G1-2): a full-horizon NPV carries its window in its name, so the total is
        # `capital_charge_total_full_horizon_npv` here. The annual block's `components` keys come from
        # extract_planning_diagnostics and are left as they are; the assertion maps the renamed key back.
        _ANNUAL_KEY = {"capital_charge_total_full_horizon_npv": "capital_charge_total_npv"}
        for _k, _v in _terms.items():
            _ak = _ANNUAL_KEY.get(_k, _k)
            assert round(_v, 0) == cost_decomposition["components"][_ak], (_k, _v, cost_decomposition["components"][_ak])
        _sum_terms = (_terms["gas_opex_npv"] + _terms["capital_charge_total_full_horizon_npv"] + _terms["selfgen_npv"]
                      + _terms["voll_penalty_npv"] + _terms["carbon_cost_npv"] + _terms["om_cost_npv"] - _terms["salvage_npv"])
        _sum_capital_lines = sum(_terms[k] for k in (
            "gas_capital_charge_npv", "public_solar_capital_charge_npv", "eaas_solar_capital_charge_npv",
            "storage_capital_charge_npv", "rebuild_capital_charge_npv", "conversion_capital_charge_npv"))
        cost_decomposition["components_unrounded"] = _terms
        cost_decomposition["reconciliation"] = {
            "npv_total_cost_usd": npv_total_cost_usd,
            "sum_of_terms_usd": _sum_terms,
            "remainder_usd": npv_total_cost_usd - _sum_terms,
            "sum_of_six_capital_lines_usd": _sum_capital_lines,
            "capital_charge_total_minus_six_lines_full_horizon_usd": _terms["capital_charge_total_full_horizon_npv"] - _sum_capital_lines,
        }

    # ------------------------------------------------------------
    # Save summary + diagnostics
    # ------------------------------------------------------------
    summary = {
        "cap_scenario": "baseline_no_policy_SLICED",
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
        # task_capex_reporting_field, Part 2: the financing headline the
        # reporting rule names (docs/frozen_state_2026_09_10.md §8) --
        # 2025-2045, gross, salvage excluded -- and its full-2025-2054-
        # horizon companion. Both from extract_planning_diagnostics()'s
        # _capex_at(), unchanged; this runner adds no new arithmetic.
        "npv_gross_capex_report_window_usd": npv_gross_capex_report_window_usd,
        "npv_gross_capex_full_horizon_usd": npv_gross_capex_full_horizon_usd,
        # TASK 53 2.4: the objective is ONE optimisation over the whole horizon, discounted to
        # the first model year; it has no 2045 counterpart (the 2025-2045 analogue of capital is
        # npv_gross_capex_report_window_usd above; the horizon-2045 objective is a separate solve,
        # B-3's cell (2045, 15)). The window is recorded rather than put in a name used by ~86 readers.
        "npv_total_cost_window": f"{int(years[0])}-{int(years[-1])}, discounted to {int(years[0])}",
        **_period,
        # Phase 2.8: TOTAL gas-side output (fleet + converted); components
        # split out so the backcast figure is not read as fleet-only.
        "backcast_gas_2025_twh_e": gas_2025,
        "backcast_gas_2025_fleet_twh_e": gas_fleet_2025,
        "backcast_gas_2025_converted_twh_e": gas_conv_2025,
        "concessional_volume_by_vintage_usd": concessional_volume_by_vintage_usd,
        "concessional_share_by_vintage": concessional_share_by_vintage,
        # Phase 2.8: per-tier conversion outcome ({} under conv_none).
        "conversion": _conv_diag,
        "leakage": leak,
        "notes": "Sliced baseline (6 slices). No emissions cap (emissions_cap=None, no emissions_constraint built).",
        # docs/output_naming.md: the FULL tag, since the filename below
        # carries only its hash8.
        "arm_tag": tag,
        # the discount rate the objective used (0.08 unless --social-discount-rate overrode it)
        "social_discount_rate_used": float(scenario["social_discount_rate"]),
        # TASK 65/66 sensitivity: present only when --gas-vintage-dispatch is given (the default summary is unchanged)
        **({"conc_share_rule": {k: v for k, v in scenario["conc_share_rule"].items() if k != "reference"}, "concessional_envelope_usd": float(scenario["concessional_envelope_usd"])} if args.conc_share_rule is not None else {}),
        **({"hydro_om_term": True, "hydro_fom_usd_per_kw_yr": float(scenario["hydro_fom_usd_per_kw_yr"])} if args.hydro_om_term else {}),
        **({"selfgen_busbar_factor_used": float(scenario["selfgen_busbar_factor"])} if args.selfgen_busbar_factor is not None else {}),
        **({"gas_vintage_dispatch": True, "gas_new_efficiency_used": float(scenario["gas_new_efficiency"]),
            "gas_eta_used": float(scenario["gas_eta"])} if gvd else {}),
    }
    with open(RESULTS_DIR / f"summary__{hash8}.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)

    # ------------------------------------------------------------
    # Firm-capacity constraint diagnostic (2.6, F0 Step 2.3): slack (MW)
    # and dual per year on m.firm_capacity_constraint.
    # ------------------------------------------------------------
    firm_capacity = firm_capacity_diagnostic(m, years)
    print("=" * 55)
    print("FIRM CAPACITY CONSTRAINT -- slack (MW) and dual by year")
    binding_years = 0
    for y in years:
        y = int(y)
        rec = firm_capacity[y]
        if abs(rec["slack_mw"]) < 1.0:
            binding_years += 1
        if y == 2025 or y == 2054 or (y - 2025) % 5 == 0:
            print(f"  {y}: slack={rec['slack_mw']:12.2f} MW   dual={rec['dual']}")
    print(f"Binding years (|slack_mw| < 1.0): {binding_years} / {len(years)}")
    print("=" * 55)

    diag_out = {
        "gas_generation_twh_e_by_year": annual_series("gas_generation"),
        "gas_capacity_mw_by_year": {
            int(y): float(pyo.value(m.gas_capacity_mw[t])) for t, y in enumerate(years)},
        "gas_add_mw_by_year": {
            int(y): float(pyo.value(m.gas_add[t])) for t, y in enumerate(years)},
        "solar_generation_twh_e_by_year": annual_series("solar_generation"),
        "hydro_generation_twh_e_by_year": annual_series("hydro_gen"),
        "storage_discharge_twh_e_by_year": annual_series("storage_discharge"),
        "storage_charge_twh_e_by_year": annual_series("storage_charge"),
        "unserved_twh_by_year": annual_series("unserved"),
        "gas_to_power_twh_th_by_year": annual_series("gas_to_power"),
        "emissions_tco2_by_year": emissions_by_year,
        "gas_shadow_pv_usd_per_twh_th_by_year": gas_shadow,
        "gas_shadow_yr_usd_per_twh_th_by_year": gas_shadow_yr,
        "slice_detail_2025": slice_2025,
        # L15: RENAMED from "slice_detail_2045" -- the data was always the
        # terminal model year (2054), never 2045; the key name predates
        # the horizon extension. Output-schema change: nothing outside
        # this file reads the old key (docs/task_hydro_gate_bucketA.md,
        # Part 1 established this; re-confirmed here before renaming).
        "slice_detail_terminal_year": slice_terminal_year,
        "selfgen_twh_by_year": annual_series("selfgen"),
        "firm_capacity": firm_capacity,
        # TASK 60: cost decomposition by term (built above, after the solve; read-only).
        "cost_decomposition": cost_decomposition,
        # TASK 65/66 sensitivity: thermal draw of the added-capacity stream (TWh_th/yr); only under the flag
        **({"gas_to_power_new_twh_th_by_year": {
                int(y): sum(float(pyo.value(m.gas_to_power_new[t, sp])) for sp in m.S)
                for t, y in enumerate(years)}} if gvd else {}),
        # docs/output_naming.md: the FULL tag, since the filename below
        # carries only its hash8.
        "arm_tag": tag,
    }
    with open(RESULTS_DIR / f"diagnostics__{hash8}.json", "w") as f:
        json.dump(json_safe(diag_out), f, indent=2)

    # ------------------------------------------------------------
    # Yearly timeseries (annual aggregates + annual capacity)
    # ------------------------------------------------------------
    ts = pd.DataFrame({
        "year": [int(y) for y in years],
        "gas_to_power_twh_th": [annual_series("gas_to_power")[int(y)] for y in years],
        "gas_generation_twh_e": [annual_series("gas_generation")[int(y)] for y in years],
        "solar_generation_twh_e": [annual_series("solar_generation")[int(y)] for y in years],
        "hydro_generation_twh_e": [annual_series("hydro_gen")[int(y)] for y in years],
        "storage_discharge_twh_e": [annual_series("storage_discharge")[int(y)] for y in years],
        "storage_charge_twh_e": [annual_series("storage_charge")[int(y)] for y in years],
        "unserved_twh": [annual_series("unserved")[int(y)] for y in years],
        "emissions_tco2": [emissions_by_year[int(y)] for y in years],
        "gas_shadow_pv_usd_per_twh_th": [gas_shadow[int(y)] for y in years],
        "gas_shadow_yr_usd_per_twh_th": [gas_shadow_yr[int(y)] for y in years],
        "solar_capacity_mw": [float(pyo.value(m.solar_capacity_mw[t])) for t in range(len(years))],
        "gas_capacity_mw": [float(pyo.value(m.gas_capacity_mw[t])) for t in range(len(years))],
        "storage_capacity_mwh": [float(pyo.value(m.storage_capacity_mwh[t])) for t in range(len(years))],
        "selfgen_twh": [annual_series("selfgen")[int(y)] for y in years],

    })
    # docs/output_naming.md: FULL tag as a header comment, since the
    # filename carries only its hash8.
    with open(RESULTS_DIR / f"timeseries__{hash8}.csv", "w") as f:
        f.write(f"# arm_tag: {tag}\n")
        ts.to_csv(f, index=False)

    # Per-slice CSV for the two inspection years. L15: the second label
    # was hardcoded "2045" (stale since the horizon extension -- the data
    # is the terminal model year, 2054); read dynamically now, as the
    # summary's "storage {int(years[-1])}" print already does above.
    slice_rows = []
    for label, rows in (("2025", slice_2025), (str(years[-1]), slice_terminal_year)):
        for r in rows:
            r2 = dict(r); r2["year"] = label
            slice_rows.append(r2)
    with open(RESULTS_DIR / f"slice_detail__{hash8}.csv", "w") as f:
        f.write(f"# arm_tag: {tag}\n")
        pd.DataFrame(slice_rows).to_csv(f, index=False)

    print("--- Sliced baseline saved ---")
    print("Solar addition (MW/year):", dv["solar_add_mw_by_year"])
    print(f"Storage capacity {int(years[-1])} (MWh):", dv[f"storage_capacity_{int(years[-1])}_mwh"])


if __name__ == "__main__":
    main()
