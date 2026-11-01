import argparse
import json
import numpy as np
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.cli_arms import add_solar_build_args, solar_build_kwargs, resolve_results_dir, complete_arm_tag  # TASK 46
from src.io import (load_econ, load_solar_capex_by_year)
from src.scenarios import (
    load_scenario, asset_lifetime_sweep, gas_deliverability_scenarios,
    conversion_scenarios, hydro_case_scenarios, tariff_scenarios, om_scenarios,
    MODEL_END_YEAR,
)
from src.optimize_model import solve_model, leakage_diagnostic
from src.optimize_model_sliced import build_model_sliced, conversion_diagnostic
from src.optimize_experiments import extract_planning_diagnostics
from src.output_naming import short_tag, hash8_of, append_index, om_tag_suffix
from src.utils import json_safe

# ============================================================
# PATH SETUP
# ============================================================


RESULTS_DIR = ROOT / "results" / "ndc"
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


_ARM_TAG_IN_TAG = {"peak_multiple_case","storage_cc_case","gas_build_case","new_gas_deliverability","gas_availability_case","gas_retirement_case","gas_capex_case","ccgt_efficiency_case","capital_recovery_case","dscr_case","capital_treatment_case","concessional_envelope_case","gas_deliverability_case","conversion_case","hydro_case","om_case","capital_case"}


def case_arm_tag(c, voll_case, args):
    """The full arm tag whose hash8 names case c's result directory (short_tag(tag, c)). One definition, used by
    the run and by --print-case-dirs, so a driver can resolve the directory names without a solve."""
    arm_tag = (
        f"{c}_{voll_case}_{args.peak_case}_{args.storage_cc_case}"
        f"_{args.gas_build_case}_{args.new_gas_deliverability}"
        f"_{args.gas_availability_case}_{args.gas_retirement_case}"
        f"_{args.gas_capex_case}_{args.ccgt_efficiency_case}"
        f"_{args.capital_recovery_case}_{args.dscr_case}_{args.capital_treatment_case}"
        f"_{args.concessional_envelope_case}_{args.gas_deliverability_case}"
        f"_{args.conversion_case}_{args.hydro_case}"
        + om_tag_suffix(args.om_case, "_")
    )
    # TASK 53 2.5 (G7-4): append the arms the tag omits (reserve_margin, tariff_case, us_inflation, window, storage).
    # TASK 83 A.1: under --capital-case-override the capital case is NOT implied by the cap-case name, so it is appended to the tag
    # (and the result gets its own hash8); without the flag the tag is unchanged.
    return complete_arm_tag(arm_tag, args, sep="_",
                            in_tag=(_ARM_TAG_IN_TAG - {"capital_case"}) if getattr(args, "capital_case", None) else _ARM_TAG_IN_TAG)


def parse_args():
    parser = argparse.ArgumentParser(description="02_run_ndc_caps: public-only NDC cap solves")
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
             "Conversion dispatch is slice-resolved and lives in "
             "build_model_sliced() only -- the annual build_model() raises "
             "on any available arm.",
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
        "--include-decline-arm",
        action="store_true",
        default=False,
        help="Opt-in: also solve 'ndc3_conditional_decline' (plan 4.2 "
             "cap-shape sensitivity -- linear decline from the ndc3_conditional "
             "2035 cap toward net-zero-2060, continued through the buffer). "
             "Default False: a default invocation solves exactly the "
             "original four NDC cases, unchanged. This arm is additive, "
             "never central, and is never included in run_central_sweep.sh.",
    )
    parser.add_argument(
        "--cases", default=None,
        help="TASK 82 B.2: comma-separated subset of the cap cases to solve (e.g. ndc3_conditional). Default None: "
             "every case the invocation would otherwise solve, unchanged. A name that is not one of those cases "
             "is an error. The arm tag of a case does not depend on this flag.")
    parser.add_argument(
        "--capital-case-override", dest="capital_case", default=None,
        help="TASK 83 A.1: replace the per-cap public-capital case (tight for unconditional, moderate for conditional) with this one for "
             "every case solved, e.g. 'unconstrained' to measure the cap without an envelope. Default None: unchanged. The case is "
             "appended to the arm tag, so these results never share a directory with the envelope-carrying ones.")
    parser.add_argument(
        "--print-case-dirs", action="store_true",
        help="print `<case> <directory name>` for every case this invocation would solve and exit (no build, no "
             "solve, nothing written). Lets run_phase_c.sh resolve the post-epoch directory names.")
    add_solar_build_args(parser)  # TASK 46: opt-in; absent = inherited default
    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def load_annual_caps(scenario_name: str, years: list[int]) -> list[float]:
    """
    Load annual emissions caps from the processed cap file, aligned to model years.
    """
    if not CAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CAP_PATH}. Run scripts/00_build_emissions_cap.py first."
        )

    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].copy()
    df = df.sort_values("year")

    # Align exactly to model years
    df = df[df["year"].isin([int(y) for y in years])]
    df = df.sort_values("year")

    caps = df["cap_tco2"].astype(float).tolist()

    if len(caps) != len(years):
        raise ValueError(
            f"Cap length {len(caps)} != number of model years {len(years)}. "
            f"Check missing years in emissions_cap.csv for scenario={scenario_name}."
        )

    return caps


# ============================================================
# RUNNERS
# ============================================================

def run_case(cap_scenario_name: str, scenario: dict, econ: dict) -> dict:
    years = [int(y) for y in scenario["years"]]
    caps = load_annual_caps(cap_scenario_name, years)
    
    # Load time-varying solar CAPEX from NREL ATB (solar_low scenario).
    # solar_low declines from $1,456k/MW (2025) to $603k/MW (2045).
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=2025, end_year=MODEL_END_YEAR,
    )
    # solar_min_build_mw_per_year override to solar_min_build_default()
    # REMOVED (2.11, Step 1) -- [UNSOURCED] (plan 2.5); load_scenario()'s
    # own 0.0 default now applies (see docs/phase_2_11_minbuild_retirement.md).

    m = build_model_sliced(
        scenario=scenario,
        econ=econ,
        emissions_cap_by_year=caps,  # annual cap trajectory
        solar_capex_by_year=solar_capex_tv,
    )

    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"Optimization failed for {cap_scenario_name}: {status}")

    # task_audit_fixes_p2 (audit finding A21): pass the SAME solar_capex_tv
    # the model was just built with, rather than letting
    # extract_planning_diagnostics() resolve its own, independent copy via
    # scenario.get("solar_capex_scenario", "solar_low") -- one resolution,
    # one source, so the model and its own diagnostic cannot silently
    # diverge if either default ever changes.
    diag = extract_planning_diagnostics(m, scenario, econ, solar_capex_series=solar_capex_tv)
    npv_total_cost_usd = float(pyo.value(m.system_cost_npv))
    cumulative_unserved_twh = sum(diag["unserved_twh_by_year"].values())
    # Leakage guard (2.6, D15). Cap is grid-only; if load is dumped to diesel
    # the NDC is "met" on paper while national emissions rise. A flagged run
    # is not an NDC-compliant plan and must not be reported as one.
    leak = leakage_diagnostic(m)
    if leak["flagged"]:
        print(f"  LEAKAGE FLAG [{cap_scenario_name}]: {leak['leakage_share']:.1%} "
              f"of emissions ({leak['leakage_tco2']/1e6:.1f} MtCO2) outside the "
              f"cap boundary. NOT NDC-compliant.")
    avg_gas_shadow_pv = np.mean(
        [v for v in diag["gas_shadow_pv_usd_per_twh_th_by_year"].values()
         if v is not None]
    )
    avg_gas_shadow_yr = np.mean(
        [v for v in diag["gas_shadow_yr_usd_per_twh_th_by_year"].values()
         if v is not None]
    )
    solar_total_built_mw = sum(
        float(pyo.value(m.solar_public_add[t]))
        + float(pyo.value(m.solar_eaas_add[t]))
        for t in range(len(years))
    )

    out = {
        "cap_scenario": cap_scenario_name,
        "decision_variables": {
            "solar_add_mw_by_year": {
                int(y):
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                for t, y in enumerate(years)
                },
                "storage_capacity_mwh":
                    float(pyo.value(m.storage_capacity_mwh[len(years)-1])),
                "solar_total_built_mw": solar_total_built_mw,
            },
        "npv_total_cost_usd": npv_total_cost_usd,
        "cumulative_unserved_twh": cumulative_unserved_twh,
        "actual_emissions_tco2_total": float(pyo.value(m.emissions)),
        "avg_gas_shadow_pv_usd_per_twh_th": avg_gas_shadow_pv,
        "avg_gas_shadow_yr_usd_per_twh_th": avg_gas_shadow_yr,
        "diagnostics": diag,
        "leakage": leak,
        # Phase 2.8: per-tier conversion outcome ({} under conv_none).
        "conversion": conversion_diagnostic(m, scenario["years"], scenario),
        }

    print("Storage discharge (TWh by year):", diag["storage_discharge_twh_e_by_year"])
    # binding constraint by year:", diag["storage_binding_by_year"])

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()
    if args.print_case_dirs:
        cases = ["ndc2_unconditional", "ndc2_conditional", "ndc3_unconditional", "ndc3_conditional"]
        if args.include_decline_arm:
            cases.append("ndc3_conditional_decline")
        for c in cases:
            print(f"{c} {short_tag(case_arm_tag(c, 'voll_mid', args), c)}")
        return
    # TASK 46: opt-in design flags. With neither given, nothing below changes.
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args, keyed=False)
    _solar_kw = solar_build_kwargs(args)

    CANONICAL_VOLL = "voll_mid"
    CANONICAL_GENSET = "genset_central"
    # ----------------------------------------------------------------
    # NDC 3.0 scenario definitions
    #
    # Each case carries two parameters that differ between unconditional
    # and conditional:
    #
    #   capital_case — public capital ceiling passed to load_scenario().
    #     unconditional: "tight" (0.50 × B* = 4.55B USD, current B* =
    #       9.104B -- task_audit_fixes_p3, A47/A29 corrected this comment,
    #       which previously said "moderate" (0.85 x a RETIRED B* =
    #       6.13B, i.e. 5.2B) and misled 15_run_pol1_ndc_comparison.py's
    #       own NDC_CONFIGS into copying the wrong tier for years)
    #       -> domestically self-funded floor, budget is binding.
    #     conditional:   "moderate" (0.85 × B* = 7.74B USD)
    #       -> international finance supplements domestic capital;
    #         apportioned power-sector share of $270bn is ~$19.8bn,
    #         moderate (7.74B) sits within that envelope while keeping
    #         the constraint economically meaningful (not unconstrained).
    #         Source: NDC 3.0 p.21, same 12.7% electricity-gen share
    #         used to derive the emissions cap.
    #
    # Note: 02_run_ndc_caps.py has NO EaaS. Public capital is the only
    # deployment channel, so capital_case is the dominant constraint.
    # ----------------------------------------------------------------
    NDC_CASES = {
        "ndc2_unconditional": {"capital_case": "tight"},
        "ndc2_conditional": {"capital_case": "moderate"},
        "ndc3_unconditional": {"capital_case": "tight"},
        "ndc3_conditional":   {"capital_case": "moderate"},
    }

    if args.include_decline_arm:
        # Plan 4.2 cap-shape sensitivity, additive and opt-in only (Step 5:
        # a default invocation must still solve exactly the original four
        # cases above, unmodified). capital_case follows ndc3_conditional's
        # ("moderate") because this arm shares its 2025-2035 caps and its
        # NDC ambition tier exactly -- only the post-2035 cap SHAPE differs.
        NDC_CASES["ndc3_conditional_decline"] = {"capital_case": "moderate"}

    if args.cases is not None:
        _want = [x for x in args.cases.split(",") if x]
        _bad = [x for x in _want if x not in NDC_CASES]
        if _bad or not _want:
            raise SystemExit(f"--cases: {_bad or args.cases!r} not among {list(NDC_CASES)}")
        NDC_CASES = {c: NDC_CASES[c] for c in _want}

    for c, cfg in NDC_CASES.items():
        scenario = load_scenario(
            **_solar_kw,
            demand_level_case="served",
            demand_case="organic_central",
            gas_deliverability_case=args.gas_deliverability_case,
            capital_case=args.capital_case or cfg["capital_case"],
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

        voll_cases = [CANONICAL_VOLL]

        for voll_case in voll_cases:

            print(f"\nRunning {c} with {voll_case}")
            econ = load_econ(voll_case, genset_case=CANONICAL_GENSET)
            out = run_case(c, scenario, econ)

            # Save outputs. docs/output_naming.md: the full arm tag below
            # exceeds the filesystem's ~255-byte per-path-component limit
            # once enough required arms accumulate (case_dir.mkdir()
            # raised OSError [Errno 36] on this sweep). The directory
            # name uses a short deterministic hash8 instead (pattern A);
            # the FULL tag is preserved as "arm_tag" in every JSON and as
            # a header comment in the CSV, and recorded in
            # RESULTS_DIR/INDEX.md so the directory stays navigable.
            arm_tag = case_arm_tag(c, voll_case, args)
            hash8 = hash8_of(arm_tag)
            append_index(RESULTS_DIR, hash8, arm_tag)
            case_dir = RESULTS_DIR / short_tag(arm_tag, c)
            case_dir.mkdir(parents=True, exist_ok=True)

            with open(case_dir / "summary.json", "w") as f:
                json.dump(
                    {
                        "cap_scenario": c,
                        "capital_case": args.capital_case or cfg["capital_case"],
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
                        # docs/tariff_arm.md, Step 4: recorded for auditability
                        # even though this runner has no EaaS channel
                        # (financing_regime='traditional' -> m.eaas_disable
                        # fixes solar_eaas_add[t] <= 0), making the arm inert.
                        # NOT appended to the output tag (deliberate
                        # deviation, step 2).
                        "tariff_case": args.tariff_case,
                        # task_om_implementation: recorded for auditability,
                        # like tariff_case above -- NOT appended to the tag.
                        "om_case": args.om_case,
                        "conversion": out["conversion"],
                        "voll_case": voll_case,
                        "voll_value_usd_per_twh": econ["UNSERVED_ENERGY_PENALTY"],
                        "decision_variables": out["decision_variables"],
                        "npv_total_cost_usd": out["npv_total_cost_usd"],
                        "cumulative_unserved_twh": out["cumulative_unserved_twh"],
                        "actual_emissions_tco2_total": out["actual_emissions_tco2_total"],
                        # docs/output_naming.md: the FULL tag, since
                        # case_dir's name carries only its hash8.
                        "arm_tag": arm_tag,
                    },
                    f,
                    indent=2,
                )

            with open(case_dir / "diagnostics.json", "w") as f:
                json.dump(
                    {"arm_tag": arm_tag, **json_safe(out["diagnostics"])},
                    f, indent=2,
                )

            # ------------------------------------------------------------
            # Save timeseries (csv)
            # ------------------------------------------------------------
            years = [int(y) for y in scenario["years"]]
            diag = out["diagnostics"]

            ts = pd.DataFrame(
                {
                    "year": years,
                    "demand_twh": [diag["demand_twh_by_year"][y] for y in years],
                    "gas_avail_twh_th": [diag["gas_avail_twh_th_by_year"][y] for y in years],
                    "gas_to_power_twh_th": [diag["fleet_gas_to_power_twh_th_by_year"][y] for y in years],
                    "gas_generation_twh_e": [diag["fleet_gas_generation_twh_e_by_year"][y] for y in years],
                    "solar_generation_twh_e": [diag["solar_generation_twh_e_by_year"][y] for y in years],
                    "storage_discharge_twh_e": [diag["storage_discharge_twh_e_by_year"][y] for y in years],
                    "unserved_twh": [diag["unserved_twh_by_year"][y] for y in years],
                    "emissions_tco2": [diag["emissions_tco2_by_year"][y] for y in years],
                    "gas_shadow_pv_usd_per_twh_th": [diag["gas_shadow_pv_usd_per_twh_th_by_year"][y] for y in years],
                    "gas_shadow_yr_usd_per_twh_th": [diag["gas_shadow_yr_usd_per_twh_th_by_year"][y] for y in years],
                    "carbon_shadow_pv_usd_per_tco2": [diag["carbon_shadow_pv_usd_per_tco2_by_year"][y] for y in years],
                    "carbon_shadow_yr_usd_per_tco2": [diag["carbon_shadow_yr_usd_per_tco2_by_year"][y] for y in years],
                    "discount_factor": [diag["discount_factor_by_year"][y] for y in years],
                    "subsidy_per_mw_usd": [diag["subsidy_per_mw_usd_by_year"][y] for y in years],
                }
            )

            # docs/output_naming.md: FULL tag as a header comment.
            with open(case_dir / "timeseries.csv", "w") as f:
                f.write(f"# arm_tag: {arm_tag}\n")
                ts.to_csv(f, index=False)

            print(f"--- {c} saved ---")
            print("Solar addition (MW/year):", out["decision_variables"]["solar_add_mw_by_year"])
            print("Storage capacity (MWh):", out["decision_variables"]["storage_capacity_mwh"])
            print("Total emissions (MtCO2):", out["actual_emissions_tco2_total"] / 1e6)
            print("Unserved 2025 (TWh):", out["diagnostics"]["unserved_twh_by_year"][2025])


if __name__ == "__main__":
    main()
