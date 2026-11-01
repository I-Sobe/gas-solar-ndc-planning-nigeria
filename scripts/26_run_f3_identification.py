"""
26_run_f3_identification.py -- F3: lexicographic identification of EaaS
                               solar capacity under an NDC cap
=====================================================================

QUESTION
--------
Is total EaaS solar capacity UNIQUELY identified at the optimal system
cost, or is there a range of EaaS deployments all achieving the same
cost (a degenerate optimum)? The F0 audit found pre-F1 degeneracy
(every H2 tie cell was baseline_no_policy). F1 was expected to resolve
it. This measures whether it did.

METHOD (per configuration, per tolerance)
-----------------------------------------
  1. Solve normally               -> Z* = value(m.system_cost_npv)
  2. Add  m.system_cost_npv <= Z* * (1 + tol);  minimise sum solar_eaas_add
  3. Same constraint;             maximise      sum solar_eaas_add
The gap (max - min) is the identification interval at that tolerance.
Reported at tol in {1e-6, 1e-5, 1e-4} (plus the required --tol value) in
ONE invocation: a degenerate LP shows a wide interval even at 1e-6; a
numerically-tight optimum widens only as tol grows.

The per-year min/max envelope is ALSO reported: a tight total with a
wide per-year profile is a different result and is not hidden.

TWO CONFIGURATIONS, REPORTED AS A PAIR (run one per invocation via
--capital-case, combine in docs/f3_identification.md):
  - public budget CONSTRAINED   (--capital-case moderate  -> 0.85 B*)
  - public budget UNCONSTRAINED (--capital-case unconstrained -> None,
    reproducing 03_run_ndc_eaas.py's public_solar_budget_npv = None)

Under F1, public and EaaS solar are priced IDENTICALLY (same unit capex,
same CRF). If public solar is unbounded the optimiser is indifferent to
the public/EaaS split and F3 would report an interval from zero to the
bankability limit -- a restatement of the configuration, not a
measurement. EaaS's claimed mechanism is that it DECOUPLES solar from
the public capital ceiling; remove the ceiling and there is nothing to
decouple from. If the unconstrained interval is wide and the
constrained one tight, THAT is the finding: EaaS capacity is identified
only where the public ceiling binds.

SCOPED CONDITIONS (this result is conditional on all five)
---------------------------------------------------------
  1. SUPERSEDED (docs/tariff_arm.md, Steps 1-3): the tariff is no longer
     a bare literal. It is now a REQUIRED --tariff-case arm threaded into
     load_scenario(tariff_case=...). --tariff-case tariff_legacy_95m
     reproduces the original 95,000,000 USD/TWh exactly; --tariff-case
     tariff_central is the sourced 95 USD/MWh NBET-negotiation midpoint
     (numerically identical, sourced provenance). The F3 interval still
     moves with the tariff -- run the sweep across arms to see it.
  2. EaaS and public solar are PHYSICALLY IDENTICAL in the current
     model. Two sourced-but-unimplemented corrections oppose each
     other: [DIST-2] behind-the-meter generation avoids the 7% TLF
     (favours EaaS); [DIST-5] distributed commercial PV carries a CAPEX
     premium +16% (2025) -> +37% (2045) per NREL ATB plus a distributed
     capacity factor differing from utility-scale 0.20 (disfavours
     EaaS). Their absence makes the two channels harder to distinguish
     and therefore makes degeneracy MORE likely, not less.
  3. capital_envelope_scenarios() carries a STALE-DOC warning: an
     earlier docstring quoted B* ~ 6.13 bn against the coded 9.104 bn,
     unreconciled. The constrained arm inherits that.
  4. gas_avail_nameplate = 0.80 against a measured 30.6% (P2-9; 29.5% before TASK 73).
     Regression-consistent; retained to avoid confounding F3.
  5. The NDC cap is an additional constraint on the same variables, so
     it will tend to NARROW the interval. A tight interval here
     establishes identification WHERE THE CAP BINDS, not in general.

CENTRAL CONFIGURATION (see TASK.md)
----------------------------------
  --capital-treatment-case f1_tranche   --capital-recovery-case cr_central
  --dscr-case dscr_central               --concessional-envelope-case env_dares
  --conversion-case conv_central         --hydro-case hydro_flat
  --gas-deliverability-case baseline     --peak-case peak_unsuppressed
  --storage-cc-case storage_cc_central   --gas-build-case gas_unconstrained
  --new-gas-deliverability constrained   --gas-retirement-case gas_retire_vintage_30
  --gas-availability-case gas_avail_nameplate
  --gas-capex-case gas_capex_central     --ccgt-efficiency-case ccgt_eff_central
  --us-inflation 0.020                   --tariff-case tariff_central
NDC case: ndc3_conditional only.   Capital cases to run: moderate, unconstrained.

The "do not change the tariff literal or make it an arm" instruction
above has since been REVERSED by docs/tariff_arm.md. Do not implement
[DIST-2]/[DIST-5].
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.append(str(ROOT))

from src.cli_arms import add_solar_build_args, solar_build_kwargs, resolve_results_dir, complete_arm_tag  # TASK 46
from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import (
    load_scenario, gas_deliverability_scenarios, conversion_scenarios,
    hydro_case_scenarios, capital_envelope_scenarios, tariff_scenarios,
    MODEL_END_YEAR,
)
from src.optimize_model import solve_model, leakage_diagnostic
from src.optimize_experiments import annual_emissions_nested
from src.optimize_model_sliced import build_model_sliced
from src.output_naming import short_tag, hash8_of, append_index
from src.utils import json_safe


RESULTS_DIR = ROOT / "results" / "f3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

NDC_CAP_SCENARIO = "ndc3_conditional"   # TASK.md: ndc3_conditional only
# TASK 48 (A.2/U4): this runner has no --om-case flag and used to hard-code the literal
# inside the load_scenario call, recording it nowhere, so no f3 summary could say which O&M
# arm produced it.  One constant now feeds both the call and the summary.
OM_CASE = "om_central"

# Tolerances always evaluated, so tolerance sensitivity is visible in
# every run (TASK.md step 1). --tol is additionally REQUIRED, forcing an
# explicit choice; its value is unioned in below.
TOLS_ALWAYS = (1e-6, 1e-5, 1e-4)

# docs/tariff_arm.md, Steps 1-3: the bare 95_000_000.0 literal ("no arm")
# is retired. The EaaS tariff now comes from a REQUIRED --tariff-case CLI
# flag threaded into load_scenario(tariff_case=...). tariff_legacy_95m
# reproduces the old literal exactly (UNSOURCED, attribution only) so
# every pre-existing F3 result stays attributable; tariff_central is the
# sourced 95 USD/MWh NBET-negotiation midpoint (numerically identical).
TARIFF_CASE_CHOICES = tuple(tariff_scenarios())

# --- choice tuples: mirror 03_run_ndc_eaas.py exactly -----------------
PEAK_CASE_CHOICES = ("peak_suppressed", "peak_unsuppressed", "peak_nirp", "peak_high")
STORAGE_CC_CASE_CHOICES = ("storage_cc_low", "storage_cc_central", "storage_cc_high")
GAS_BUILD_CASE_CHOICES = ("gas_ndc_17gw", "gas_unconstrained", "gas_none")
NEW_GAS_DELIVERABILITY_CHOICES = ("constrained", "exempt")
GAS_AVAILABILITY_CASE_CHOICES = ("gas_avail_observed", "gas_avail_central", "gas_avail_nameplate")
GAS_RETIREMENT_CASE_CHOICES = ("gas_retire_none", "gas_retire_proxy", "gas_retire_vintage_30", "gas_retire_vintage_40", "gas_retire_vintage_50")
GAS_CAPEX_CASE_CHOICES = ("gas_capex_low", "gas_capex_central", "gas_capex_high")
CCGT_EFFICIENCY_CASE_CHOICES = ("ccgt_eff_iso", "ccgt_eff_central", "ccgt_eff_low")
CAPITAL_RECOVERY_CASE_CHOICES = ("cr_central", "cr_ida21", "cr_comm_low", "cr_comm_high")
DSCR_CASE_CHOICES = ("dscr_regression", "dscr_central", "dscr_low", "dscr_high")
CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")
CONCESSIONAL_ENVELOPE_CASE_CHOICES = ("env_dares", "env_dares_plus", "env_none", "env_5x", "env_10x")
GAS_DELIVERABILITY_CASE_CHOICES = tuple(gas_deliverability_scenarios().keys())
CONVERSION_CASE_CHOICES = tuple(conversion_scenarios().keys())
HYDRO_CASE_CHOICES = tuple(hydro_case_scenarios())

# --capital-case: keys of capital_envelope_scenarios() (which already
# contains the literal "unconstrained" -> None) plus that literal
# explicitly, per TASK.md. deduped, order preserved.
CAPITAL_CASE_CHOICES = tuple(dict.fromkeys(list(capital_envelope_scenarios()) + ["unconstrained"]))


def parse_args():
    p = argparse.ArgumentParser(
        description="26_run_f3_identification: lexicographic identification interval for EaaS solar MW"
    )
    p.add_argument(
        "--capital-case",
        required=True,
        choices=CAPITAL_CASE_CHOICES,
        help="Public solar capital envelope fed to load_scenario(capital_case=...). "
             "No default. 'moderate' = 0.85 x B* (constrained arm); 'unconstrained' "
             "= public_solar_budget_npv = None (reproduces 03_run_ndc_eaas.py). "
             "F3 does NOT override public_solar_budget_npv post-call -- the whole "
             "point is to keep whichever ceiling this arm implies.",
    )
    p.add_argument(
        "--tol",
        required=True,
        type=float,
        help="Objective-slack tolerance for the lexicographic re-solve "
             "(m.system_cost_npv <= Z*(1+tol)). No default: an unstated slack "
             "would size the identification interval silently. The runner ALSO "
             "always evaluates 1e-6, 1e-5, 1e-4 in the same invocation so "
             "tolerance sensitivity is visible; this value is unioned in.",
    )
    p.add_argument("--peak-case", required=True, choices=PEAK_CASE_CHOICES)
    p.add_argument("--storage-cc-case", required=True, choices=STORAGE_CC_CASE_CHOICES)
    p.add_argument("--gas-build-case", required=True, choices=GAS_BUILD_CASE_CHOICES)
    p.add_argument("--new-gas-deliverability", required=True, choices=NEW_GAS_DELIVERABILITY_CHOICES)
    p.add_argument("--gas-availability-case", required=True, choices=GAS_AVAILABILITY_CASE_CHOICES)
    p.add_argument("--gas-retirement-case", required=True, choices=GAS_RETIREMENT_CASE_CHOICES)
    p.add_argument("--gas-capex-case", required=True, choices=GAS_CAPEX_CASE_CHOICES)
    p.add_argument("--ccgt-efficiency-case", required=True, choices=CCGT_EFFICIENCY_CASE_CHOICES)
    p.add_argument("--capital-recovery-case", required=True, choices=CAPITAL_RECOVERY_CASE_CHOICES)
    p.add_argument("--dscr-case", required=True, choices=DSCR_CASE_CHOICES)
    p.add_argument("--us-inflation", required=True, type=float)
    p.add_argument("--capital-treatment-case", required=True, choices=CAPITAL_TREATMENT_CASE_CHOICES)
    p.add_argument("--concessional-envelope-case", required=True, choices=CONCESSIONAL_ENVELOPE_CASE_CHOICES)
    p.add_argument("--gas-deliverability-case", required=True, choices=GAS_DELIVERABILITY_CASE_CHOICES)
    p.add_argument("--conversion-case", required=True, choices=CONVERSION_CASE_CHOICES)
    p.add_argument("--hydro-case", required=True, choices=HYDRO_CASE_CHOICES)
    p.add_argument("--tariff-case", required=True, choices=TARIFF_CASE_CHOICES,
                   help="EaaS service tariff arm fed to "
                        "load_scenario(tariff_case=...). No default: it "
                        "parameterises m.eaas_bankability, which gates F-27. "
                        "tariff_legacy_95m = the retired bare literal.")
    add_solar_build_args(p)  # TASK 46: opt-in; absent = inherited default
    return p.parse_args()


def load_annual_caps(scenario_name: str, years: list[int]) -> list[float]:
    if not CAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CAP_PATH}. Run scripts/00_build_emissions_cap.py first."
        )
    df = pd.read_csv(CAP_PATH)
    df = df[df["scenario"] == scenario_name].copy()
    df = df.sort_values("year")
    df = df[df["year"].isin([int(y) for y in years])]
    df = df.sort_values("year")
    caps = df["cap_tco2"].astype(float).tolist()
    if len(caps) != len(years):
        raise ValueError(
            f"Cap length {len(caps)} != number of model years {len(years)} "
            f"for scenario={scenario_name}."
        )
    return caps


def build_f3_model(args, econ):
    """load_scenario() call + post-call mutations, mirroring 03_run_ndc_eaas.py."""
    scenario = load_scenario(
        **solar_build_kwargs(args),  # TASK 46: {} unless --solar-build-case given
        demand_level_case="served",
        demand_case="organic_central",
        gas_deliverability_case=args.gas_deliverability_case,
        capital_case=args.capital_case,
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
        om_case=OM_CASE,
    )

    # Post-call mutation: financing_regime ONLY. The tariff now comes from
    # load_scenario(tariff_case=args.tariff_case) above -- the bare
    # 95_000_000.0 literal formerly set here is retired (docs/tariff_arm.md,
    # Steps 1-3). F3 deliberately does NOT set
    # scenario["public_solar_budget_npv"] = None -- the constrained arm
    # must keep its 0.85 B* ceiling; the unconstrained arm already gets
    # None from load_scenario(capital_case="unconstrained").
    scenario["financing_regime"] = "eaas"

    years = [int(y) for y in scenario["years"]]
    caps = load_annual_caps(NDC_CAP_SCENARIO, years)
    solar_capex_tv = load_solar_capex_by_year(
        scenario_name="solar_low",
        start_year=int(years[0]),
        end_year=int(years[-1]),
    )

    m = build_model_sliced(
        scenario=scenario,
        econ=econ,
        emissions_cap_by_year=caps,
        solar_capex_by_year=solar_capex_tv,
    )
    return scenario, m, years


def total_eaas_mw(m, T):
    return float(sum(pyo.value(m.solar_eaas_add[t]) for t in T))


def eaas_profile(m, T):
    return [float(pyo.value(m.solar_eaas_add[t])) for t in T]


def full_solution_vector(m, T, years):
    """
    The same quantities normal_solve already captures, plus the fields
    needed to tell apart "public solar absorbs the difference" from
    "total solar collapses and another technology meets demand" (see
    docs/f3_identification.md, "Full solution vector"). Read from `m` at
    whatever solution it currently holds -- no re-solve, no new
    Expression added to the model. Every field below is an EXISTING Var
    or Expression on the sliced builder; call this immediately after a
    solve and before the next objective swap.
    """
    t2045 = years.index(2045)
    t2054 = years.index(2054)
    out = {
        "solar_public_add_mw_total":
            float(sum(pyo.value(m.solar_public_add[t]) for t in T)),
        "solar_capacity_mw_2045": float(pyo.value(m.solar_capacity_mw[t2045])),
        "gas_capacity_mw_2045": float(pyo.value(m.gas_capacity_mw[t2045])),
        "storage_capacity_mwh_2054": float(pyo.value(m.storage_capacity_mwh[t2054])),
        "selfgen_twh_e_total":
            float(sum(pyo.value(m.selfgen[t, sp]) for t in T for sp in m.S)),
        "gas_generation_twh_e_total":
            float(sum(pyo.value(m.gas_generation[t, sp]) for t in T for sp in m.S)),
    }
    # Converted generation only exists on the model when a conversion arm
    # is active (hasattr guard, not a new Expression) -- None otherwise,
    # not zero, so a reader cannot mistake "no conversion mechanism" for
    # "conversion mechanism present but idle."
    if hasattr(m, "convert_generation"):
        out["converted_generation_twh_e_total"] = float(sum(
            pyo.value(m.convert_generation[tier, t, sp])
            for tier in ("A", "B", "C") for t in T for sp in m.S
        ))
    else:
        out["converted_generation_twh_e_total"] = None
    return out


def bankability_slack_per_mw_by_year(m, scenario, years):
    """
    Per-year slack in m.eaas_bankability, USD per MW of EaaS solar
    (docs/tariff_arm.md, Step 3). No such reader existed before this task.

        slack_per_mw[t] = tariff * solar_energy_per_mw * remaining_npv_factor[t]
                          / required_margin
                          - (capex[t] - eaas_subsidy[t] / solar_eaas_add[t])

    Positive ⇒ the tariff more than covers the effective private cost at
    year t (constraint slack, EaaS not tariff-limited there); ~zero ⇒ it
    binds. Guard: solar_eaas_add[t] == 0 ⇒ the constraint is 0 <= 0 and
    carries no information — emit None, not zero (1e-6 MW absorbs solver
    noise). Evaluated at whatever solution `m` currently holds.
    """
    tariff = scenario["solar_service_tariff_usd_per_twh"]
    solar_energy_per_mw = scenario["solar_cf"] * 8760 / 1_000_000
    required_margin = scenario["required_margin"]

    out = {}
    for t, y in enumerate(years):
        eaas_add = float(pyo.value(m.solar_eaas_add[t]))
        if eaas_add <= 1e-6:
            out[int(y)] = None
            continue
        rnf = float(pyo.value(m.remaining_npv_factor[t]))
        capex = float(pyo.value(m.solar_capex_param[t]))
        subsidy = float(pyo.value(m.eaas_subsidy[t]))
        bankable_revenue_per_mw = tariff * solar_energy_per_mw * rnf / required_margin
        effective_private_cost_per_mw = capex - subsidy / eaas_add
        out[int(y)] = bankable_revenue_per_mw - effective_private_cost_per_mw
    return out


def reoptimise_eaas(m, scenario, T, years, Zstar, tol, sense):
    """Fix objective at Z*(1+tol), then optimise total EaaS MW in `sense`."""
    for name in ("f3_zstar_con", "f3_eaas_obj"):
        if m.component(name) is not None:
            m.del_component(name)

    m.objective.deactivate()
    m.f3_zstar_con = pyo.Constraint(
        expr=m.system_cost_npv <= Zstar * (1.0 + tol)
    )
    m.f3_eaas_obj = pyo.Objective(
        expr=pyo.quicksum(m.solar_eaas_add[t] for t in T), sense=sense
    )

    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"F3 re-solve failed (sense={sense}, tol={tol}): {status}")

    # Read the full solution vector off THIS solve, immediately, before
    # the objective/constraint swap below is undone for the next call
    # (docs/f3_identification.md, "Full solution vector"). No re-solve.
    out = {
        "total_eaas_mw": total_eaas_mw(m, T),
        "eaas_add_mw_by_year": eaas_profile(m, T),
        "system_cost_npv": float(pyo.value(m.system_cost_npv)),
    }
    out.update(full_solution_vector(m, T, years))

    # restore the model to its canonical state for the next call
    m.del_component("f3_eaas_obj")
    m.del_component("f3_zstar_con")
    m.objective.activate()
    return out


def main():
    args = parse_args()
    # TASK 46: opt-in design flags; with neither given nothing changes.
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args, keyed=False)

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)
    scenario, m, years = build_f3_model(args, econ)
    T = range(len(years))

    public_budget_npv = scenario.get("public_solar_budget_npv", None)
    constrained = public_budget_npv is not None
    has_pbc = bool(getattr(m, "has_public_budget_constraint", False))
    assert constrained == has_pbc, (
        f"public_solar_budget_npv={public_budget_npv} but "
        f"has_public_budget_constraint={has_pbc} -- inconsistent."
    )

    # ---- Solve 1: normal ------------------------------------------------
    status = solve_model(m, scenario=scenario)
    if not status["optimal"]:
        raise RuntimeError(f"F3 normal solve failed: {status}")
    Zstar = float(pyo.value(m.system_cost_npv))
    # TASK 80 C: the leakage guard (2.6, D3) on the NORMAL solve, before the min/max re-optimisations perturb the model; the
    # summaries stored neither self-generation nor emissions, which cannot be recovered afterwards.
    leak_normal = leakage_diagnostic(m)
    emis_normal = annual_emissions_nested(m, years)
    normal = {
        "total_eaas_mw": total_eaas_mw(m, T),
        "eaas_add_mw_by_year": eaas_profile(m, T),
        "solar_public_add_mw_total": float(sum(pyo.value(m.solar_public_add[t]) for t in T)),
    }
    # Step 3 (docs/tariff_arm.md): per-year bankability slack, read off the
    # NORMAL solve (before the F3 min/max re-optimisations perturb the
    # model). Without it, EaaS deployment unchanged across tariff arms
    # cannot be told apart from a constraint that never binds.
    normal_bankability_slack_per_mw = bankability_slack_per_mw_by_year(
        m, scenario, years
    )

    tols = sorted(set(TOLS_ALWAYS) | {float(args.tol)})

    by_tol = {}
    for tol in tols:
        lo = reoptimise_eaas(m, scenario, T, years, Zstar, tol, pyo.minimize)
        hi = reoptimise_eaas(m, scenario, T, years, Zstar, tol, pyo.maximize)

        min_prof = lo["eaas_add_mw_by_year"]
        max_prof = hi["eaas_add_mw_by_year"]
        # per-year envelope: elementwise across the min-total and
        # max-total solutions (TASK.md: three solves per configuration,
        # so the envelope is read off those, not re-optimised per year).
        env_lo = [min(a, b) for a, b in zip(min_prof, max_prof)]
        env_hi = [max(a, b) for a, b in zip(min_prof, max_prof)]

        by_tol[f"{tol:.0e}"] = {
            "tol": tol,
            "zstar_ceiling_usd": Zstar * (1.0 + tol),
            "total_eaas_mw_min": lo["total_eaas_mw"],
            "total_eaas_mw_max": hi["total_eaas_mw"],
            "identification_interval_mw": hi["total_eaas_mw"] - lo["total_eaas_mw"],
            "min_solution": lo,
            "max_solution": hi,
            "per_year_envelope": {
                "year": years,
                "eaas_add_mw_lower": env_lo,
                "eaas_add_mw_upper": env_hi,
                "eaas_add_mw_span": [h - l for l, h in zip(env_lo, env_hi)],
                "max_per_year_span_mw": max(h - l for l, h in zip(env_lo, env_hi)),
            },
        }

    summary = {
        "finding": "F3",
        "leakage": leak_normal,
        "emissions_by_year_normal_solve": emis_normal,
        "question": "Is total EaaS solar capacity uniquely identified at the optimal system cost?",
        "ndc_cap_scenario": NDC_CAP_SCENARIO,
        "capital_case": args.capital_case,
        "public_budget_constrained": constrained,
        "public_solar_budget_npv": public_budget_npv,
        "has_public_budget_constraint": has_pbc,
        "tariff_case": args.tariff_case,
        "om_case": OM_CASE,
        "eaas_tariff_usd_per_twh": scenario["solar_service_tariff_usd_per_twh"],
        "financing_regime": scenario["financing_regime"],
        "required_margin": scenario.get("required_margin"),
        "years": years,
        "Zstar_system_cost_npv_usd": Zstar,
        "normal_solve": normal,
        "bankability_slack_per_mw_by_year": normal_bankability_slack_per_mw,
        "tolerances_evaluated": tols,
        "tol_arg": float(args.tol),
        "by_tol": by_tol,
        "arm_tag": None,  # filled in below once the full tag is built
        "arms": {
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
            "tariff_case": args.tariff_case,
            "voll_case": CANONICAL_VOLL,
            "genset_case": CANONICAL_GENSET,
        },
    }

    # FULL arm tag -- preserved in summary.json (where it is actually
    # read from), NOT used as a path component. Sixteen required arms
    # push this past the filesystem's ~255-byte per-component limit, so
    # the directory name is a short deterministic digest instead (F3
    # addendum; see docs/f3_identification.md "Path-length fix").
    arm_tag = (
        f"{NDC_CAP_SCENARIO}_{args.capital_case}_{CANONICAL_VOLL}"
        f"_{args.peak_case}_{args.storage_cc_case}_{args.gas_build_case}"
        f"_{args.new_gas_deliverability}_{args.gas_availability_case}"
        f"_{args.gas_retirement_case}_{args.gas_capex_case}_{args.ccgt_efficiency_case}"
        f"_{args.capital_recovery_case}_{args.dscr_case}_{args.capital_treatment_case}"
        f"_{args.concessional_envelope_case}_{args.gas_deliverability_case}"
        f"_{args.conversion_case}_{args.hydro_case}_{args.tariff_case}"
    )
    # TASK 53 2.5 (G7-4): append the arms the tag omits (om_case, reserve_margin, us_inflation, window, storage).
    arm_tag = complete_arm_tag(arm_tag, args, sep="_", in_tag={"peak_multiple_case","storage_cc_case","gas_build_case","new_gas_deliverability","gas_availability_case","gas_retirement_case","gas_capex_case","ccgt_efficiency_case","capital_recovery_case","dscr_case","capital_treatment_case","concessional_envelope_case","gas_deliverability_case","conversion_case","hydro_case","tariff_case","capital_case"})
    summary["arm_tag"] = arm_tag

    # docs/output_naming.md: this hash8/short_name/INDEX.md scheme is now
    # the shared src/output_naming.py helper -- refactored onto it here
    # since short_tag()'s hash computation is byte-identical to the
    # inline version this replaced (same sha1, same [:8] slice, same
    # f"{prefix}_{hash8}" shape with prefix=f"{NDC_CAP_SCENARIO}_
    # {args.capital_case}"), so no existing hash changes.
    hash8 = hash8_of(arm_tag)
    short_name = short_tag(arm_tag, f"{NDC_CAP_SCENARIO}_{args.capital_case}")
    case_dir = RESULTS_DIR / short_name
    # Deterministic: re-running the same configuration overwrites.
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)

    # Append (never overwrite) a hash8 -> full arm tag line so results/f3/
    # is navigable without opening each JSON.
    append_index(RESULTS_DIR, hash8, arm_tag)

    print(f"\n=== F3 [{args.capital_case}] ===")
    print(f"public budget constrained: {constrained}  (npv={public_budget_npv})")
    print(f"Z* = {Zstar:,.0f} USD")
    for k, v in by_tol.items():
        print(
            f"  tol {k}: EaaS total MW in "
            f"[{v['total_eaas_mw_min']:,.1f}, {v['total_eaas_mw_max']:,.1f}]  "
            f"interval={v['identification_interval_mw']:,.1f} MW  "
            f"max per-year span={v['per_year_envelope']['max_per_year_span_mw']:,.1f} MW"
        )
    print(f"arm_tag: {arm_tag}")
    print(f"saved -> {case_dir / 'summary.json'}  (hash8={hash8})")


if __name__ == "__main__":
    main()
