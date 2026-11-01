"""
04_run_fin2_tariff_sweep.py  —  FIN-2 Tariff Bankability Sweep
===============================================================

RESEARCH QUESTION (FIN-2)
--------------------------
What is the minimum EaaS tariff at which private capital becomes
self-financing, what subsidy (eaas_subsidy) does the model require below
that tariff, and how does the threshold shift with capital structure
(blended WACC, via --concessional-envelope) and finance conditionality
(NDC cap scenario)?

The self-financing threshold T* is analytically:

    T* = SOLAR_CAPEX_PER_MW × required_margin / npv_energy_per_mw

where npv_energy_per_mw is the horizon-discounted TWh per MW, both of
which depend on the blended WACC (see src/blended_finance.py).

This script solves the full NDC-constrained optimisation across a tariff
grid, for every blended-finance capital structure in
blended_finance_scenarios(), crossed with every NDC scenario in NDC_CASES.

NOTE ON THE "EMPIRICAL" CROSSING
----------------------------------
run_fin2_blended_bankability_sweep() (src/optimize_experiments.py) sets
eaas_subsidy equal to the financing gap at every tariff. That means the
tariff at which eaas_subsidy_npv hits zero — subsidy_zero_crossing in the
summary below — coincides with the analytical T* BY CONSTRUCTION, not as
an independent empirical discovery. Treat subsidy_zero_crossing as a
consistency check on the LP/analytical wiring, not as empirical
confirmation of T*.

WHAT THIS ADDS OVER EXISTING RUNS
-----------------------------------
The canonical runs in 03_run_ndc_eaas.py use tariff=95M USD/TWh, which
is well above T* for both scenarios. This sweep varies the tariff from
below T* to above T* to:
  1. Check that the LP's subsidy_zero_crossing lines up with the
     analytical T* (a consistency check — see note above)
  2. Quantify the subsidy required at each below-threshold tariff
  3. Show that concessional finance (lower blended WACC / conditional NDC
     cap) lowers T* — the direct modelled effect of concessional finance

OUTPUTS
--------
  All filenames are tagged with the --concessional-envelope arm and
  --deployable-fraction (output_tag = "{arm}__df{fraction}") so a second
  arm does not overwrite the first arm's results:

  results/fin2/tariff_sweep_results__{output_tag}.csv   — one row per
      (ndc_scenario, blend_scenario, tariff); every field the sweep
      function + this script attach, including zero_subsidy_reason.
  results/fin2/tariff_sweep_results__{output_tag}.json  — same rows, JSON.
  results/fin2/tariff_sweep_summary__{output_tag}.json  — threshold
      crossings and key stats, collapsed per (ndc_scenario, blend_scenario).

CLI
----
  --concessional-envelope {dares_only,dares_plus,unbounded}  (default: unbounded)
      Level-2 concessional capital scarcity envelope (see
      concessional_envelope_npv() in src/blended_finance.py).
      dares_only = 750M USD facility, dares_plus = 1050M USD facility,
      unbounded = no scarcity constraint (Level 1: rate only).
  --deployable-fraction FLOAT  (default: 1.0)
      Fraction of the facility deployable to grid-connected EaaS solar.

RUN ORDER
----------
  00_build_emissions_cap.py  (must exist)
  01_run_baseline.py         (must exist)
  04_run_fin2_tariff_sweep.py  ← this script
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.cli_arms import add_solar_build_args, solar_build_kwargs, resolve_results_dir  # TASK 46
from src.io import load_econ
from src.scenarios import (
    load_scenario,
    TARIFF_SWEEP_GRID,
    MODEL_END_YEAR,
    blended_finance_scenarios,
)
from src.optimize_experiments import run_fin2_blended_bankability_sweep
from src.blended_finance import concessional_envelope_npv
from src.cli_arms import add_standard_arms, standard_arms_kwargs

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = ROOT / "results" / "fin2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"

# Canonical VoLL for all FIN-2 runs — consistent with 02 and 03.
CANONICAL_VOLL = "voll_mid"

# Concessional envelope arms: total facility size (USD, NPV) fed to
# concessional_envelope_npv() in src/blended_finance.py. "unbounded" skips
# the scarcity constraint entirely (Level 1: rate only, no envelope).
CONCESSIONAL_ENVELOPE_ARMS = {
    "dares_only": 750_000_000.0,
    "dares_plus": 1_050_000_000.0,
    "unbounded": None,
}

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
# NOT the same thing as this script's own --concessional-envelope flag
# below (dares_only/dares_plus/unbounded): that is the PRE-EXISTING,
# NPV-discounted, EaaS-solar-only Level-2 scarcity constraint, now RETIRED
# from both builders (docs/f1_implementation.md, "Envelope resolution") --
# its CLI flag and CONCESSIONAL_ENVELOPE_ARMS dict are left in this script
# unchanged but now feed an inert scenario key (see the report). This new
# flag, --concessional-envelope-case, is F1's OWN, separate mechanism.
CONCESSIONAL_ENVELOPE_CASE_CHOICES = ("env_dares", "env_dares_plus", "env_none", "env_5x", "env_10x")


def parse_args():
    parser = argparse.ArgumentParser(description="FIN-2 tariff bankability sweep")
    parser.add_argument(
        "--concessional-envelope",
        choices=sorted(CONCESSIONAL_ENVELOPE_ARMS),
        default="unbounded",
        help="Concessional capital envelope arm (Level 2 scarcity constraint).",
    )
    parser.add_argument(
        "--deployable-fraction",
        type=float,
        default=1.0,
        help="Fraction of the facility deployable to grid-connected EaaS solar.",
    )
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
             "unconstrained and fund every channel 100%% concessionally. "
             "NOT the same as --concessional-envelope above (that flag's "
             "underlying mechanism is retired -- see module docstring).",
    )
    # docs/task_runner_repair.md, Step Two: conversion_case, hydro_case and
    # tariff_case were the three sentinel-required load_scenario() keys
    # this script omitted outright (Shape A -- confirmed no others were
    # missing before this edit). Added via the shared helper rather than
    # three more hand-copied add_argument() calls, consistent with the
    # existing 14-flag block above (itself now a candidate for a future
    # migration onto add_standard_arms(parser), out of this task's scope).
    add_standard_arms(parser, include=["conversion_case", "hydro_case", "tariff_case", "om_case"])
    add_solar_build_args(parser)  # TASK 46: opt-in; absent = inherited default
    return parser.parse_args()

# NDC scenarios to sweep — each has its own capital_case, which is the core
# of the conditional vs unconditional finance comparison.
#
# F1, Decision 2: the per-case required_margin (1.10 unconditional / 1.05
# conditional) formerly used here as a proxy for concessional vs commercial
# cost of capital is RETIRED as a double-count. That distinction is now
# carried directly by capital_recovery_case's r_conc/r_comm CRF rates;
# required_margin is purely the DSCR coverage cushion
# (load_scenario(dscr_case=...)), held constant across all NDC_CASES here.
NDC_CASES = {
    # Robustness arm: NDC 2.0 ( BAU-relative, no apportionment required)
    "ndc2_unconditional": {
        "capital_case": "tight",
        "ndc_cap_scenario": "ndc2_unconditional",
    },
    "ndc2_conditional": {
        "capital_case": "moderate",
        "ndc_cap_scenario": "ndc2_conditional",
    },
    "baseline_no_policy": {
        "capital_case":    "unconstrained",
        "ndc_cap_scenario": None,
    },
    "ndc3_unconditional": {
        "capital_case":    "tight",
        "ndc_cap_scenario": "ndc3_unconditional",
    },
    "ndc3_conditional": {
        "capital_case":    "moderate",
        "ndc_cap_scenario": "ndc3_conditional",
    },
}


# ============================================================
# MAIN
# ============================================================
CANONICAL_GENSET = "genset_central"
def main():

    args = parse_args()
    # TASK 46: opt-in design flags. With neither given, nothing below changes.
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)
    _solar_kw = solar_build_kwargs(args)
    envelope_arm = args.concessional_envelope
    total_facility_usd = CONCESSIONAL_ENVELOPE_ARMS[envelope_arm]
    envelope_npv = (
        None if total_facility_usd is None
        else concessional_envelope_npv(total_facility_usd, args.deployable_fraction)
    )
    envelope_npv_str = "None" if envelope_npv is None else f"{envelope_npv/1e6:.1f}M USD"
    print(f"Concessional envelope arm: {envelope_arm}  "
          f"(envelope_npv={envelope_npv_str}, deployable_fraction={args.deployable_fraction})")

    # Every output file this script writes is tagged with the arm + deployable
    # fraction -- otherwise a second --concessional-envelope run silently
    # overwrites the first arm's results/summary.
    output_tag = (
        f"{envelope_arm}__df{args.deployable_fraction:g}__{args.peak_case}"
        f"__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}"
        f"__{args.gas_availability_case}__{args.gas_retirement_case}"
        f"__{args.gas_capex_case}__{args.ccgt_efficiency_case}"
        f"__{args.capital_recovery_case}__{args.dscr_case}__{args.capital_treatment_case}"
        f"__{args.concessional_envelope_case}"
    )

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    all_rows = []

    for ndc_name, cfg in NDC_CASES.items():

        print(f"\n{'='*60}")
        print(f"  FIN-2 sweep: {ndc_name}")
        print(f"  capital_case={cfg['capital_case']}")
        print(f"  tariff levels: {[t//1_000_000 for t in TARIFF_SWEEP_GRID]}M USD/TWh")
        print(f"{'='*60}")

        scenario = load_scenario(
            **_solar_kw,
            demand_level_case="served",
            demand_case="organic_central",
            gas_deliverability_case="baseline",
            capital_case=cfg["capital_case"],
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

        # Activate EaaS — tariff will be overwritten per point in the sweep.
        # F1, Decision 2: no per-case required_margin override here anymore --
        # required_margin is sourced solely from load_scenario(dscr_case=...)
        # above, constant across all NDC_CASES.
        scenario["financing_regime"] = "eaas"

        rows = run_fin2_blended_bankability_sweep(
            base_scenario=scenario,
            econ=econ,
            tariff_grid=TARIFF_SWEEP_GRID,
            blend_scenarios=blended_finance_scenarios(),
            ndc_cap_scenario=cfg["ndc_cap_scenario"],
            builder="sliced",  # A5 addendum, Part 2: was hardwired to the annual builder
            concessional_envelope_npv=envelope_npv,
            cap_path=str(CAP_PATH),
        )

        for r in rows:
            r["ndc_scenario"] = ndc_name
            r["concessional_envelope_arm"] = envelope_arm

            t_m = r["tariff_m_usd_per_twh"]
            sf = r["is_self_financing"]
            sub = r.get("eaas_subsidy_npv_usd", float("nan"))
            status = r["status"]
            print(
                f"  blend={r['blend_scenario']:<18}"
                f"tariff={t_m:5.0f}M  "
                f"self-financing={str(sf):5}  "
                f"subsidy_npv={sub/1e6:8.2f}M  "
                f"[{status}]"
            )

        all_rows.extend(rows)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / f"tariff_sweep_results__{output_tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Persist the full per-(blend, tariff) row list, not just the summary --
    # the summary block below collapses each blend to a handful of scalars.
    rows_json_path = RESULTS_DIR / f"tariff_sweep_results__{output_tag}.json"
    with open(rows_json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved: {rows_json_path}")

    # ── Summary: threshold crossing per (NDC scenario, blend) ──────────────────
    # Each blend has its own blended WACC and therefore its own analytical T*,
    # so the threshold is no longer a single value per NDC scenario.
    summary = {}
    for ndc_name in NDC_CASES:
        subset = df[df["ndc_scenario"] == ndc_name]
        summary[ndc_name] = {}

        for blend_label in subset["blend_scenario"].unique():
            blend_subset = subset[subset["blend_scenario"] == blend_label]
            optimal = blend_subset[blend_subset["status"] == "optimal"]

            # Analytical threshold (same for all rows within a blend)
            threshold = optimal["threshold_usd_per_twh"].iloc[0] if len(optimal) else None
            blended_wacc = optimal["blended_wacc"].iloc[0] if len(optimal) else None

            # Zero-subsidy crossing: lowest tariff where eaas_subsidy_npv_usd == 0
            # (or effectively zero — below 1000 USD NPV). This coincides with the
            # analytical T* BY CONSTRUCTION, since the LP sets eaas_subsidy equal
            # to the financing gap at every tariff (see
            # run_fin2_blended_bankability_sweep in src/optimize_experiments.py).
            # It is a consistency check on the LP/analytical wiring, NOT an
            # empirical confirmation of T*.
            self_fin = optimal[optimal["eaas_subsidy_npv_usd"] < 1000.0]
            subsidy_zero_crossing = (
                float(self_fin["tariff_usd_per_twh"].min())
                if len(self_fin) > 0
                else None
            )

            # Max subsidy (at lowest tariff)
            max_subsidy_row = (
                optimal.loc[optimal["eaas_subsidy_npv_usd"].idxmax()].to_dict()
                if len(optimal) > 0 else {}
            )

            summary[ndc_name][blend_label] = {
                "dscr_case":                     args.dscr_case,
                "concessional_envelope_case":    args.concessional_envelope_case,
                "blended_wacc":                  blended_wacc,
                "analytical_threshold_usd_per_twh":  threshold,
                "analytical_threshold_m_usd_per_twh": threshold / 1e6 if threshold else None,
                "subsidy_zero_crossing_usd_per_twh":   subsidy_zero_crossing,
                "subsidy_zero_crossing_m_usd_per_twh": subsidy_zero_crossing / 1e6 if subsidy_zero_crossing else None,
                "max_subsidy_npv_usd":          max_subsidy_row.get("eaas_subsidy_npv_usd"),
                "max_subsidy_at_tariff_m":      max_subsidy_row.get("tariff_m_usd_per_twh"),
            }

            print(f"\n{ndc_name} / {blend_label} (wacc={blended_wacc:.3f}):" if blended_wacc is not None
                  else f"\n{ndc_name} / {blend_label}:")
            print(f"  Analytical T*:          {threshold/1e6:.3f} M USD/TWh" if threshold else "  T*: N/A")
            print(f"  Subsidy zero-crossing:  {subsidy_zero_crossing/1e6:.3f} M USD/TWh (consistency check, not empirical confirmation)" if subsidy_zero_crossing else "  Subsidy zero-crossing: not crossed in sweep")
            print(f"  Max subsidy NPV: {max_subsidy_row.get('eaas_subsidy_npv_usd', 0)/1e6:.2f} M USD "
                  f"at tariff={max_subsidy_row.get('tariff_m_usd_per_twh', '?')}M")

    json_path = RESULTS_DIR / f"tariff_sweep_summary__{output_tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    print("\n=== FIN-2 sweep complete ===")
    print("Key thesis results to report (blended_central):")
    headline_vals = {}
    for ndc_name, blends in summary.items():
        s = blends.get("blended_central", {})
        t_star = s.get("analytical_threshold_m_usd_per_twh")
        headline_vals[ndc_name] = t_star
        print(f"  {ndc_name}: T* = {f'{t_star:.2f}' if t_star is not None else 'N/A'} M USD/TWh  ")
    if "ndc3_conditional" in headline_vals and "ndc3_unconditional" in headline_vals:
        v1, v2 = headline_vals["ndc3_conditional"], headline_vals["ndc3_unconditional"]
        if v1 is not None and v2 is not None:
            delta = abs(v1 - v2)
            print(f"\n  Delta T* (conditional vs unconditional, blended_central) = {delta:.3f} M USD/TWh")
            print("  This is the quantified effect of concessional NDC finance")
            print("  on private EaaS bankability.")


if __name__ == "__main__":
    main()
