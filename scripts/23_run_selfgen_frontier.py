"""
Phase 2.6 -- Step E: grid service obligation frontier.

EPS AXIS: see docs/eps_axis_rationale.md, which builds on this script's
own calibration below to give the REL suite one documented rationale
for why its scripts legitimately sweep different eps ranges.

WHAT THIS REPLACES
------------------
The first version of this script compared VoLL-priced against genset-priced
cost on an UNCONSTRAINED tight-capital case. That was miscalibrated: with no
bound on off-grid supply the LP routed 569 TWh through selfgen (more than
total horizon demand), and the resulting -94.6% objective delta measured the
size of the VoLL assumption, not the economics of self-generation. Any backstop
priced below VoLL would have produced the same figure.

WHAT THIS DOES INSTEAD
----------------------
Sweeps the GRID SERVICE OBLIGATION -- the bound on (selfgen + unserved) as a
fraction of Tier 1 demand -- and reports the cost of the shortfall at each
service level, priced two ways.

CALIBRATION OF THE SWEEP AXIS
-----------------------------
reliability_levels_log() returns [0.20 ... 0.05], i.e. grid-served floors of
80-95%. That axis is infeasible almost everywhere here and tells you nothing.
The correct range comes from the model's own supply bound: Tier 1 demand
reaches ~81 TWh by 2045 while grid supply is bounded near 40 TWh, so the
projected shortfall fraction is of order 0.4-0.5. The sweep therefore runs
0.05 to 0.60.

NOTE: an earlier calibration argument compared this fraction against NIRP's
lambda (0.51-0.57). That comparison was WRONG -- lambda is grid / (grid +
selfgen) on a Tier 2 base, while epsilon is shortfall / Tier 1 demand.
Different denominators. The conclusion (widen the axis) survives; the reasoning
above replaces it.

WHY MODE = 'annual' ONLY
------------------------
A service obligation is a commitment to customers in each year. Under a
monotonically rising deficit, 'total' lets early slack subsidise a late-horizon
collapse -- the system could serve 95% through the 2030s and 30% in the 2050s
and still report compliance. The mode comparison belongs in
09b_run_rel1_mode_sensitivity.py, which already sweeps both modes and can be
re-run post-2.6 against its own pre-2.6 output.

WHY PRICING DOES NOT MOVE THE FRONTIER
--------------------------------------
The obligation bounds (selfgen + unserved) irrespective of price, so the
FEASIBLE SET IS IDENTICAL under both pricings and the infeasibility threshold
is the same. Pricing changes the COST at each feasible point. The headline is
therefore the public-capex vs private-diesel-spend trade-off, not a shift in
feasibility.

WHAT unserved MEANS HERE -- READ BEFORE QUOTING ANY NUMBER
----------------------------------------------------------
Under genset pricing 'unserved' is near zero BY CONSTRUCTION, because the
backstop is cheaper and absorbs the shortfall. It is NOT evidence that demand
is met. Three quantities, three meanings:

  selfgen + unserved  grid service shortfall -- the planning constraint
  unserved alone      truly unmet demand GIVEN the backstop assumption
  selfgen cost / CO2  the welfare and emissions burden the shortfall imposes
                      on private actors

DECLARED ASSUMPTION: the backstop is unbounded and available to all. In
reality 86% of MSMEs own generators but household ownership is far lower, and
the poorest have neither grid nor generator. An unbounded backstop therefore
OVERSTATES how well the shortfall is absorbed. This belongs in the limitations
section.

TODO (separate commit): rename reliability_max_unserved_fraction ->
max_offgrid_fraction and reliability_constraint -> grid_service_obligation in
both builders. This is a service obligation, not an adequacy standard: a
household running a genset was not served by the grid, but no capacity was
short, so LOLE/EENS do not describe it. Rename is deliberately NOT bundled
with this behaviour change.

OUTPUT
------
results/selfgen_frontier/frontier.csv   -- one row per (builder, pricing, eps)
results/selfgen_frontier/frontier.json  -- same, plus run metadata
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.cli_arms import resolve_results_dir
from src.scenarios import (
    load_scenario, gas_deliverability_scenarios, conversion_scenarios,
    hydro_case_scenarios, tariff_scenarios, MODEL_END_YEAR,
)
from src.optimize_model import build_model, solve_model
from src.optimize_model_sliced import (
    build_model_sliced, annual_totals, conversion_diagnostic,
)
from src.io import load_econ, load_solar_capex_by_year
from src.utils import json_safe

FORCING_CAPITAL_CASE = "tight"
CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"
RELIABILITY_MODE = "annual"

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
# Tier port B: gas_deliverability_case was hardcoded "baseline" in
# make_scenario()'s load_scenario() call, which stranded the GMP 2026
# policy-target arms (gas_deliv_*, commit 5b811f2). Now a REQUIRED CLI
# flag, no default -- same discipline as the arms above. Choices sourced
# live from gas_deliverability_scenarios() so the historical-drift and
# GMP policy-target families cannot drift out of sync with src/scenarios.py.
GAS_DELIVERABILITY_CASE_CHOICES = tuple(gas_deliverability_scenarios().keys())
# Phase 2.8: OCGT-to-CCGT conversion arm. REQUIRED, no default. This
# script runs both builders; the annual arm raises on any available arm
# (conversion is slice-resolved), so use 'conv_none' unless only the
# sliced arm is needed. An available arm also requires f1_tranche.
CONVERSION_CASE_CHOICES = tuple(conversion_scenarios().keys())
# Correction plan 1.5: hydro trajectory arm. REQUIRED, no default.
HYDRO_CASE_CHOICES = tuple(hydro_case_scenarios())
# docs/tariff_arm.md, Steps 1-3: EaaS service tariff arm. REQUIRED, no
# default -- replaces the retired solar_tariff_case's silent 65,000,000
# USD/TWh default. Choices sourced live from tariff_scenarios().
TARIFF_CASE_CHOICES = tuple(tariff_scenarios())


def parse_args():
    parser = argparse.ArgumentParser(description="23_run_selfgen_frontier: grid service obligation frontier")
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
             "the pre-2.8 regression arm. An available arm requires "
             "f1_tranche and is slice-resolved (the annual arm of this "
             "script's dual-builder loop will raise on it).",
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

# Calibrated to the model's own projected shortfall fraction (~0.4-0.5),
# not to reliability_levels_log() which tops out at 0.20.
EPS_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
            0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

RESULTS_DIR = ROOT / "results" / "selfgen_frontier"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def genset_rates():
    """Per-TWh rates for the avoided-genset metrics, read straight from the
    processed CSV. Reported as rates because 'avoided per TWh of grid supply'
    is exactly the price, the emission factor and the fuel intensity."""
    df = pd.read_csv(ROOT / "data/cost/processed/genset_cost.csv", thousands=",")
    row = df[df["Scenario"] == CANONICAL_GENSET].iloc[0]
    eta = float(row["genset_eta_hhv"])
    ef_th = float(row["ef_th_hhv_tco2_per_mwh_th"])
    sfc = float(row["sfc_l_per_kwh"])
    return {
        "usd_per_twh": float(row["total_usd_per_twh_e"]),
        "tco2_per_twh": (ef_th / eta) * 1e6,      # EF_e DERIVED, never stored
        "litres_per_twh": sfc * 1e9,
    }


def make_scenario(peak_case, storage_cc_case, gas_build_case, new_gas_deliverability,
                   gas_availability_case, gas_retirement_case, gas_capex_case,
                   ccgt_efficiency_case, capital_recovery_case, dscr_case,
                   us_inflation, capital_treatment_case, concessional_envelope_case,
                   gas_deliverability_case, conversion_case, hydro_case, tariff_case):
    sc = load_scenario(
        demand_level_case="served",
        demand_case="organic_central",
        capital_case=FORCING_CAPITAL_CASE,
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
    return sc


def demand_from_balance(m, n, sliced):
    """Tier 1 demand read off the balance constraint's lower bound.

    Read from the constraint rather than from the scenario dict so it cannot
    drift from what the model actually enforces.
    """
    if not sliced:
        return sum(float(pyo.value(m.balance[t].lower)) for t in range(n))
    return sum(
        float(pyo.value(c.lower))
        for c in m.balance.values()
        if c.lower is not None
    )


def summarise(m, years, sliced, scenario):
    n = len(years)
    if sliced:
        tot = annual_totals(m, years)
        selfgen = sum(tot["selfgen"].values())
        unserved = sum(tot["unserved"].values())
    else:
        selfgen = sum(float(pyo.value(m.selfgen[t])) for t in range(n))
        unserved = sum(float(pyo.value(m.unserved[t])) for t in range(n))

    demand = demand_from_balance(m, n, sliced)

    public_capex_npv = sum(
        float(pyo.value(m.DF[t]))
        * float(pyo.value(m.solar_public_add[t]))
        * float(pyo.value(m.solar_capex_param[t]))
        for t in range(n)
    )

    return {
        "feasible": True,
        "objective_usd": float(pyo.value(m.system_cost_npv)),
        "demand_twh": demand,
        "selfgen_twh": selfgen,
        "unserved_twh": unserved,
        "shortfall_twh": selfgen + unserved,
        "shortfall_fraction_realised": (selfgen + unserved) / demand if demand else None,
        "grid_served_twh": demand - selfgen - unserved,
        "solar_built_mw": sum(
            float(pyo.value(m.solar_public_add[t])) + float(pyo.value(m.solar_eaas_add[t]))
            for t in range(n)
        ),
        "final_storage_mwh": float(pyo.value(m.storage_capacity_mwh[n - 1])),
        "emissions_total_tco2": float(pyo.value(m.emissions)),
        "emissions_grid_only_tco2": float(pyo.value(m.emissions_grid_only)),
        "emissions_selfgen_tco2": float(pyo.value(m.emissions_selfgen)),
        "public_solar_capex_npv_usd": public_capex_npv,
        "private_selfgen_npv_usd": float(pyo.value(m.cost_selfgen_npv)),
        "unserved_voll_npv_usd": float(pyo.value(m.cost_unserved_voll_npv)),
        # Phase 2.8: {} for the annual arm and under conv_none.
        "conversion": conversion_diagnostic(m, years, scenario) if sliced else {},
    }


def solve_point(builder, pricing, eps, sliced, peak_case, storage_cc_case,
                 gas_build_case, new_gas_deliverability, gas_availability_case,
                 gas_retirement_case, gas_capex_case, ccgt_efficiency_case,
                 capital_recovery_case, dscr_case, us_inflation, capital_treatment_case,
                 concessional_envelope_case, gas_deliverability_case, conversion_case,
                 hydro_case, tariff_case):
    sc = make_scenario(peak_case, storage_cc_case, gas_build_case, new_gas_deliverability,
                        gas_availability_case, gas_retirement_case, gas_capex_case,
                        ccgt_efficiency_case, capital_recovery_case, dscr_case,
                        us_inflation, capital_treatment_case, concessional_envelope_case,
                        gas_deliverability_case, conversion_case, hydro_case, tariff_case)
    years = sc["years"]
    capex = load_solar_capex_by_year("solar_low", int(years[0]), int(years[-1]))
    econ = (load_econ(CANONICAL_VOLL, genset_case=None) if pricing == "voll"
            else load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET))

    m = builder(
        scenario=sc, econ=econ,
        # Plan 0.4, item C: 1e18 sentinel -> None ("no cap", already the
        # builder's own guarded meaning for this argument).
        emissions_cap=None, emissions_cap_by_year=None,
        solar_capex_by_year=capex,
        # TODO rename -> max_offgrid_fraction (separate commit)
        reliability_max_unserved_fraction=eps,
        reliability_mode=RELIABILITY_MODE,
    )
    try:
        status = solve_model(m, scenario=sc)
        if not status.get("optimal", False):
            return {"feasible": False, "reason": str(status)}
    except RuntimeError as exc:
        # Infeasibility is a DATA POINT here, not an error. 09b already treats
        # it this way; the frontier is defined by where it starts occurring.
        return {"feasible": False, "reason": str(exc)[:120]}
    return summarise(m, years, sliced, sc)


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    rates = genset_rates()
    rows = []

    for builder_label, builder, sliced in (
        ("annual", build_model, False),
        ("sliced", build_model_sliced, True),
    ):
        for pricing in ("voll", "genset"):
            for eps in EPS_GRID:
                rec = solve_point(builder, pricing, eps, sliced, args.peak_case,
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
                rec.update(builder=builder_label, pricing=pricing, eps=eps)
                rows.append(rec)

    tag = f"{args.peak_case}__{args.storage_cc_case}__{args.gas_build_case}__{args.new_gas_deliverability}__{args.gas_availability_case}__{args.gas_retirement_case}__{args.gas_capex_case}__{args.ccgt_efficiency_case}__{args.capital_recovery_case}__{args.dscr_case}__{args.capital_treatment_case}__{args.concessional_envelope_case}__{args.gas_deliverability_case}__{args.conversion_case}__{args.hydro_case}"
    df = pd.DataFrame(rows)
    front_cols = ["builder", "pricing", "eps", "feasible"]
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]
    df.to_csv(RESULTS_DIR / f"frontier__{tag}.csv", index=False)

    with open(RESULTS_DIR / f"frontier__{tag}.json", "w") as f:
        json.dump(json_safe({
            "forcing_capital_case": FORCING_CAPITAL_CASE,
            "voll_case": CANONICAL_VOLL,
            "genset_case": CANONICAL_GENSET,
            "reliability_mode": RELIABILITY_MODE,
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
            # docs/tariff_arm.md, Step 4: recorded for auditability. This
            # runner never sets financing_regime (checked -- grep shows no
            # occurrence in this file), so load_scenario()'s "traditional"
            # default holds throughout and m.eaas_disable fixes
            # solar_eaas_add[t] <= 0: the arm is inert here too. NOT
            # appended to the output tag (deliberate deviation, step 2).
            "tariff_case": args.tariff_case,
            "eps_grid": EPS_GRID,
            "avoided_genset_rates_per_twh_grid_supply": rates,
            "rows": rows,
        }), f, indent=2)

    # ------------------------------------------------------------------
    # Printed report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  GRID SERVICE OBLIGATION FRONTIER -- capital_case='{FORCING_CAPITAL_CASE}', "
          f"mode='{RELIABILITY_MODE}'")
    print(f"{'=' * 78}")
    print("  eps = maximum fraction of Tier 1 demand met off-grid (selfgen + unserved).")
    print("  Lower eps = stricter obligation on the grid.\n")

    for builder_label in ("annual", "sliced"):
        print(f"\n  --- {builder_label.upper()} " + "-" * 56)
        print(f"  {'eps':>6} | {'VoLL-priced':^34} | {'genset-priced':^34}")
        print(f"  {'':>6} | {'obj bn':>9}{'short TWh':>12}{'unsrv TWh':>13} | "
              f"{'obj bn':>9}{'short TWh':>12}{'unsrv TWh':>13}")
        for eps in EPS_GRID:
            cells = []
            for pricing in ("voll", "genset"):
                r = next(x for x in rows
                         if x["builder"] == builder_label
                         and x["pricing"] == pricing and x["eps"] == eps)
                if not r["feasible"]:
                    cells.append(f"{'INFEASIBLE':>34}")
                else:
                    cells.append(f"{r['objective_usd'] / 1e9:>9.2f}"
                                 f"{r['shortfall_twh']:>12.1f}"
                                 f"{r['unserved_twh']:>13.3f}")
            print(f"  {eps:>6.2f} | {cells[0]} | {cells[1]}")

        feas = [r["eps"] for r in rows
                if r["builder"] == builder_label and r["pricing"] == "genset"
                and r["feasible"]]
        if feas:
            print(f"\n  Tightest feasible obligation: eps = {min(feas):.2f} "
                  f"(grid must serve {(1 - min(feas)):.0%} of Tier 1 demand)")
            print("  Identical under both pricings: the obligation bounds "
                  "(selfgen + unserved)")
            print("  irrespective of price, so pricing cannot move the feasible set.")
        else:
            print("\n  No feasible point on this axis. Widen EPS_GRID upward.")

    # ---- public vs private trade-off, genset pricing -----------------------
    print(f"\n{'=' * 78}")
    print("  PUBLIC CAPEX vs PRIVATE DIESEL SPEND (genset-priced, annual)")
    print(f"{'=' * 78}")
    print(f"  {'eps':>6}{'public solar capex':>22}{'private selfgen':>20}"
          f"{'selfgen CO2':>16}")
    print(f"  {'':>6}{'(bn USD, NPV)':>22}{'(bn USD, NPV)':>20}{'(MtCO2)':>16}")
    for eps in EPS_GRID:
        r = next(x for x in rows if x["builder"] == "annual"
                 and x["pricing"] == "genset" and x["eps"] == eps)
        if not r["feasible"]:
            print(f"  {eps:>6.2f}{'INFEASIBLE':>22}")
            continue
        print(f"  {eps:>6.2f}"
              f"{r['public_solar_capex_npv_usd'] / 1e9:>22.2f}"
              f"{r['private_selfgen_npv_usd'] / 1e9:>20.2f}"
              f"{r['emissions_selfgen_tco2'] / 1e6:>16.1f}")
    print("\n  This is the EaaS argument: every TWh the state does not fund is a")
    print("  TWh Nigerians buy privately, at the rates below.")

    print(f"\n  AVOIDED-GENSET RATES per TWh of grid supply ({CANONICAL_GENSET}):")
    print(f"    cost   {rates['usd_per_twh'] / 1e6:>12,.1f} m USD / TWh")
    print(f"    CO2    {rates['tco2_per_twh'] / 1e3:>12,.1f} ktCO2 / TWh")
    print(f"    diesel {rates['litres_per_twh'] / 1e6:>12,.1f} m litres / TWh")

    print(f"\n  CAUTION: under genset pricing 'unsrv TWh' is near zero BY")
    print("  CONSTRUCTION -- the backstop absorbs the shortfall. It is not")
    print("  evidence that demand is met. Quote 'short TWh' as the service")
    print("  shortfall and the selfgen cost/CO2 as the burden it imposes.")
    print(f"\n  saved: {RESULTS_DIR / f'frontier__{tag}.csv'}")
    print(f"         {RESULTS_DIR / f'frontier__{tag}.json'}\n")


if __name__ == "__main__":
    main()
