"""
Shared CLI arm-flag helper for scripts/.

WHY THIS EXISTS (register L27, docs/task_shared_fn_repair.md)
---------------------------------------------------------------
The defect this repair sequence closes -- a stale load_scenario() calling
convention that silently omitted required arms -- was replicated across six
library functions (src/optimize_experiments.py) and nine runners (this
task) for one structural reason: the 17-flag argparse block that supplies
those arms was hand-copied, file to file, rather than defined once. Every
time a new required key was added to load_scenario() (docs/methodology_
protocols.md Section 1), every copy had to be updated by hand, and several
were not.

This module is the fix for the COPYING, not the arms themselves -- it does
not change what load_scenario() requires. It provides one definition of the
standard arm flags, imported by every runner that needs them, so the next
required-key addition to load_scenario() is a one-file change to this
module rather than an N-file hand-edit that can silently miss a caller.

REFERENCE IMPLEMENTATION
-------------------------
scripts/01b_run_baseline_sliced.py's argparse block is the pattern this
module codifies: every flag required (no defaults), choices= drawn live
from the arm registries in src/scenarios.py, and help text naming the
specific incident that motivates having no default (the 65,000,000
USD/TWh silent tariff default, the unsourced 900,000 USD/MW gas capex
literal, and so on). That help text is carried over verbatim below, not
rewritten.

reserve_margin -- IS a standard arm (task_epsilon_constraint, Part 5)
-----------------------------------------------------------------------
Originally excluded, on the grounds recorded in register L9: 01b_'s own
reference implementation does not expose it as a CLI flag, passing the
hardcoded literal reserve_margin="nirp_spinning" directly at its
load_scenario() call site instead, and L9 records that as a canonical
project-wide constant, not a swept arm -- "the convention belongs at the
call site, not inside the library."

That reasoning holds for a runner with its own hand-written
load_scenario() call. It BREAKS for a runner whose entire argument
surface is add_standard_arms(parser) + **standard_arms_kwargs(args) --
nothing then supplies reserve_margin, and load_scenario() correctly
raises TypeError. Confirmed at runtime: every one of the six full-set
callers of this module (05_, 15_, 18_, 19_, 20_, 28_) failed identically
before this fix, for exactly this reason.

reserve_margin is therefore now a required flag here, accepting EITHER
the literal "nirp_spinning" OR a numeric fraction (e.g. 0.15) -- the same
two forms load_scenario() itself accepts, validated the same way (see
_reserve_margin_type below). This does not change the VALUE any existing
runner uses: every runner that already passes the literal "nirp_spinning"
(directly, or now via this flag) still resolves to "nirp_spinning" --
only runners whose entire surface is this module gain the ability to
state a different value, which none currently do.

This is the fourth instance of one defect shape in this project: a
load-bearing value excluded from a shared surface on the reasoning that
"the convention belongs at the call site" -- correct until the call site
stops existing as an independent thing and becomes wholly this module's
output. The retired 65,000,000 USD/TWh tariff default, the unsourced
storage_duration_hours constant, om_case hardcoded to "om_central" across
six library functions (register L27), and now this, are the same shape:
a value assumed to always be supplied explicitly at each site, until a
generalisation (a shared helper, in every one of these four cases)
removed the explicit site without anyone re-stating the value there.

THREE ARMS WITH NO LIVE REGISTRY: a real gap, not an oversight
-------------------------------------------------------------------
gas_build_case, new_gas_deliverability and capital_treatment_case are
required CLI flags in 01b_'s own reference implementation, but unlike
every other arm here, src/scenarios.py defines no *_scenarios() registry
function for any of the three -- only inline tuple/string checks inside
load_scenario() itself and the builders (grep "gas_ndc_17gw", "pre_f1_
overnight", "exempt" across src/*.py; confirmed no such function exists).
Their choices are therefore hardcoded tuples below, copied verbatim from
01b_'s own GAS_BUILD_CASE_CHOICES / NEW_GAS_DELIVERABILITY_CHOICES /
CAPITAL_TREATMENT_CASE_CHOICES -- the same staleness class this module
otherwise closes, but with no live source to close it against. Flagged in
docs/task_runner_repair.md rather than silently worked around; adding a
registry function for these three would be a src/scenarios.py edit,
outside this task's whitelist.

USAGE
------
    from src.cli_arms import add_standard_arms, standard_arms_kwargs

    parser = argparse.ArgumentParser(...)
    add_standard_arms(parser)                       # the full set
    add_standard_arms(parser, include=["conversion_case", "hydro_case"])
    add_standard_arms(parser, exclude=["om_case"])   # state the reason
    args = parser.parse_args()
    scenario = load_scenario(..., **standard_arms_kwargs(args))

include / exclude let a runner take a subset where a flag is genuinely
inapplicable (mutually exclusive; pass at most one). Do NOT import this
module into src/scenarios.py or either builder -- it is a CLI helper for
scripts/, nothing more.
"""

import argparse

from src.scenarios import (
    solar_build_scenarios,
    peak_multiple_scenarios,
    storage_cc_scenarios,
    gas_availability_scenarios,
    gas_retirement_scenarios,
    gas_capex_scenarios,
    ccgt_efficiency_scenarios,
    capital_recovery_scenarios,
    dscr_scenarios,
    concessional_envelope_scenarios,
    gas_deliverability_scenarios,
    conversion_scenarios,
    hydro_case_scenarios,
    tariff_scenarios,
    om_scenarios,
)

# See "THREE ARMS WITH NO LIVE REGISTRY" above -- no *_scenarios() function
# exists in src/scenarios.py for these three; copied verbatim from 01b_.
GAS_BUILD_CASE_CHOICES = ("gas_ndc_17gw", "gas_unconstrained", "gas_none")
NEW_GAS_DELIVERABILITY_CHOICES = ("constrained", "exempt")
CAPITAL_TREATMENT_CASE_CHOICES = ("pre_f1_overnight", "f1_tranche")


def _reserve_margin_type(raw):
    """
    argparse type= for --reserve-margin. A plain choices= list cannot
    express "either the literal 'nirp_spinning' or a numeric fraction" --
    this function is the validation load_scenario() itself performs
    (src/scenarios.py, reserve_margin handling), applied at parse time so
    a bad value fails at the CLI, not three solves into a sweep.

    Mirrors load_scenario()'s own two branches exactly: the string
    'nirp_spinning' is passed through unchanged; any other string is
    rejected (not silently coerced); a numeric string is parsed as a
    float and must be >= 0.
    """
    if raw == "nirp_spinning":
        return raw
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"reserve_margin must be a numeric fraction (e.g. 0.15) or "
            f"the literal 'nirp_spinning', got {raw!r}."
        )
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"reserve_margin must be >= 0, got {value}."
        )
    return value


def _arm_specs():
    """
    The standard arm table: (kwarg, flag, dest, choices, type, help).

    kwarg  : the load_scenario() keyword this flag feeds, and the key used
             by include=/exclude= and by standard_arms_kwargs()'s output.
    dest   : the argparse attribute name on the parsed args. Equal to
             kwarg for every flag except peak_multiple_case, where 01b_'s
             own reference implementation uses the flag text "--peak-case"
             (not "--peak-multiple-case") and therefore the dest "peak_case"
             -- preserved here so the flag TEXT users type matches 01b_
             exactly; standard_arms_kwargs() below bridges dest -> kwarg so
             load_scenario() still receives peak_multiple_case=....
    choices: tuple of valid values, or None for a free-form flag
             (us_inflation). Read from the LIVE registry functions above,
             not hardcoded, except the three noted at module level.
    type   : argparse type, or None (choices flags are left as str).

    A function, not a module-level constant, so every choices= tuple is
    re-read from its registry on each call -- consistent with the
    "live, not hardcoded" requirement this module exists to enforce on
    its own callers.
    """
    return [
        ("peak_multiple_case", "--peak-case", "peak_case",
         tuple(peak_multiple_scenarios().keys()), None,
         "Peak-demand shape arm fed to load_scenario(peak_multiple_case=...). "
         "No default: an unstated demand shape would build a firm-capacity "
         "constraint silently."),
        ("reserve_margin", "--reserve-margin", "reserve_margin",
         None, _reserve_margin_type,
         "Reserve margin fed to load_scenario(reserve_margin=...). Accepts "
         "a numeric fraction (e.g. 0.15) or the literal 'nirp_spinning' "
         "(derives a per-year fraction from RESERVE_MARGIN_SPINNING_MW / "
         "peak_mw[t]). No default: an unstated reserve margin would build "
         "a firm-capacity constraint against an unstated assumption with "
         "no error, warning, or visible signature in the output "
         "(load_scenario()'s own raise text, src/scenarios.py)."),
        ("storage_cc_case", "--storage-cc-case", "storage_cc_case",
         tuple(storage_cc_scenarios().keys()), None,
         "Storage firm-capacity credit arm fed to load_scenario(storage_cc_case=...). "
         "No default: an unstated storage capacity credit would keep storage the "
         "model's best firm-capacity resource on an unsourced assumption."),
        ("gas_build_case", "--gas-build-case", "gas_build_case",
         GAS_BUILD_CASE_CHOICES, None,
         "New gas capacity build policy fed to load_scenario(gas_build_case=...). "
         "No default: an unstated build policy would let new gas capacity be "
         "built (or not) silently."),
        ("new_gas_deliverability", "--new-gas-deliverability", "new_gas_deliverability",
         NEW_GAS_DELIVERABILITY_CHOICES, None,
         "New gas capacity's fuel-deliverability treatment fed to "
         "load_scenario(new_gas_deliverability=...). No default: an unstated "
         "treatment would relax (or not) the fleet fuel constraint for new "
         "build silently."),
        ("gas_availability_case", "--gas-availability-case", "gas_availability_case",
         tuple(gas_availability_scenarios().keys()), None,
         "Gas fleet availability arm fed to load_scenario(gas_availability_case=...). "
         "No default: an unstated availability figure would build both the "
         "gas annual-energy ceiling and the gas firm-capacity credit silently."),
        ("gas_retirement_case", "--gas-retirement-case", "gas_retirement_case",
         tuple(gas_retirement_scenarios().keys()), None,
         "Gas fleet retirement schedule fed to load_scenario(gas_retirement_case=...). "
         "No default: an unstated retirement schedule would retire the "
         "baseline gas fleet silently."),
        ("gas_capex_case", "--gas-capex-case", "gas_capex_case",
         tuple(gas_capex_scenarios().keys()), None,
         "New-build gas unit cost arm fed to load_scenario(gas_capex_case=...). "
         "No default: an unstated capex figure would price every new gas "
         "MW silently."),
        ("ccgt_efficiency_case", "--ccgt-efficiency-case", "ccgt_efficiency_case",
         tuple(ccgt_efficiency_scenarios().keys()), None,
         "New-build CCGT efficiency arm fed to load_scenario(ccgt_efficiency_case=...). "
         "No default: an unstated efficiency would size the exempt "
         "fuel-equivalent allowance silently."),
        ("capital_recovery_case", "--capital-recovery-case", "capital_recovery_case",
         tuple(capital_recovery_scenarios().keys()), None,
         "Per-tranche capital recovery rate/tenor arm fed to "
         "load_scenario(capital_recovery_case=...). No default: an unstated "
         "recovery period would price every tranche's CRF() charge silently."),
        ("dscr_case", "--dscr-case", "dscr_case",
         tuple(dscr_scenarios().keys()), None,
         "DSCR coverage cushion arm fed to load_scenario(dscr_case=...). "
         "No default: an unstated cushion would size the EaaS bankability "
         "constraint silently."),
        ("us_inflation", "--us-inflation", "us_inflation",
         None, float,
         "US inflation rate (fraction, e.g. 0.020) fed to "
         "load_scenario(us_inflation=...), used ONLY to deflate "
         "capital_recovery_case's nominal rates. No default."),
        ("capital_treatment_case", "--capital-treatment-case", "capital_treatment_case",
         CAPITAL_TREATMENT_CASE_CHOICES, None,
         "F1 objective-form selector fed to load_scenario(capital_treatment_case=...). "
         "'pre_f1_overnight' reproduces the pre-F1 objective; 'f1_tranche' "
         "replaces it with the per-tranche CRF annuity. No default: an "
         "unstated choice would silently pick one of two materially "
         "different objective functions."),
        ("concessional_envelope_case", "--concessional-envelope-case", "concessional_envelope_case",
         tuple(concessional_envelope_scenarios().keys()), None,
         "F1 concessional envelope arm fed to "
         "load_scenario(concessional_envelope_case=...). No default: an "
         "unstated envelope would leave concessional volume "
         "unconstrained and fund every channel 100%% concessionally."),
        ("gas_deliverability_case", "--gas-deliverability-case", "gas_deliverability_case",
         tuple(gas_deliverability_scenarios().keys()), None,
         "Gas deliverability-to-power arm fed to "
         "load_scenario(gas_deliverability_case=...). No default. Choices "
         "span two epistemic families: historical-drift trajectories "
         "(downside/baseline/upside/shock_recovery and their flat_* "
         "level-equivalents) and GMP 2026 policy targets "
         "(gas_deliv_flat / gas_deliv_gmp_partial / gas_deliv_gmp_full, "
         "the planned-increment fractions I = 0 / 0.50 / 1.0)."),
        ("conversion_case", "--conversion-case", "conversion_case",
         tuple(conversion_scenarios().keys()), None,
         "OCGT-to-CCGT conversion arm fed to "
         "load_scenario(conversion_case=...). No default. 'conv_none' is "
         "the pre-2.8 regression arm (conversion unavailable); "
         "'conv_central' / 'conv_low' / 'conv_high' set retrofit capex "
         "(USD/kW nameplate), capacity uplift and retrofit efficiency. "
         "An available arm requires --capital-treatment-case f1_tranche."),
        ("hydro_case", "--hydro-case", "hydro_case",
         tuple(hydro_case_scenarios()), None,
         "Hydro trajectory arm fed to load_scenario(hydro_case=...). "
         "No default. 'hydro_flat' is the pre-1.5 regression arm "
         "(11.47 TWh_e flat); 'hydro_committed' holds the ~12.8 TWh_e "
         "2025 committed level flat; 'hydro_ndc' ramps toward the NDC's "
         "10,400 MW by 2030 then holds flat. All exogenous -- hydro is "
         "never a decision variable."),
        ("tariff_case", "--tariff-case", "tariff_case",
         tuple(tariff_scenarios().keys()), None,
         "EaaS service tariff arm fed to load_scenario(tariff_case=...). No "
         "default: the retired solar_tariff_case defaulted to 65,000,000 "
         "USD/TWh silently, with no error, warning or visible signature in "
         "the output. Inert in runners with no EaaS channel "
         "(financing_regime='traditional' fixes solar_eaas_add to 0 via "
         "m.eaas_disable), but load_scenario() requires it regardless. See "
         "tariff_scenarios()."),
        ("om_case", "--om-case", "om_case",
         tuple(om_scenarios()), None,
         "Fixed + variable O&M arm fed to load_scenario(om_case=...). No "
         "default: the retired STORAGE_OM_PER_MWH_YR was defined in io.py "
         "but never read by either builder, for any technology, with no "
         "error or visible signature in the output. See om_scenarios()."),
    ]


def add_standard_arms(parser, *, include=None, exclude=None):
    """
    Add the standard required arm flags to an argparse parser.

    Every flag is required, with no default (matches 01b_'s discipline
    exactly) -- an omitted arm must fail argparse's own "required" check,
    not fall through to a silent value.

    include / exclude take an iterable of load_scenario() kwarg names (see
    _arm_specs()) to add only a subset -- pass at most one. A runner using
    exclude= must state the reason in its own report entry
    (docs/task_runner_repair.md), per this task's brief; this function
    itself does not enforce that, since the reason belongs in the report,
    not in code.
    """
    if include is not None and exclude is not None:
        raise ValueError("add_standard_arms: pass include or exclude, not both.")

    specs = _arm_specs()
    all_keys = {kwarg for kwarg, *_ in specs}

    if include is not None:
        unknown = set(include) - all_keys
        if unknown:
            raise ValueError(f"add_standard_arms: unknown include key(s): {sorted(unknown)}")
        specs = [s for s in specs if s[0] in include]
    elif exclude is not None:
        unknown = set(exclude) - all_keys
        if unknown:
            raise ValueError(f"add_standard_arms: unknown exclude key(s): {sorted(unknown)}")
        specs = [s for s in specs if s[0] not in exclude]

    for kwarg, flag, dest, choices, type_, help_text in specs:
        kw = dict(dest=dest, required=True, help=help_text)
        if choices is not None:
            kw["choices"] = choices
        if type_ is not None:
            kw["type"] = type_
        parser.add_argument(flag, **kw)


def standard_arms_kwargs(args):
    """
    Return the standard arms from parsed args as a load_scenario() kwargs
    dict.

    Reads whichever of _arm_specs()'s kwarg names are present as attributes
    on args -- so a runner that called add_standard_arms(parser,
    include=[...]) gets back only that subset, and one that called it with
    no include/exclude gets the full set, reserve_margin included
    (task_epsilon_constraint, Part 5 -- see module docstring). A runner
    that still passes reserve_margin as its own literal (rather than via
    this module) must NOT also request it through include=, or
    load_scenario() raises "got multiple values for keyword argument" --
    see docs/task_epsilon_constraint.md Part 5's duplicate-keyword audit.
    """
    kwargs = {}
    for kwarg, _flag, dest, _choices, _type, _help in _arm_specs():
        if hasattr(args, dest):
            kwargs[kwarg] = getattr(args, dest)
    return kwargs


# ---------------------------------------------------------------------------
# TASK 46 Part A: opt-in solar-build design flags (Phase C Group 3)
# ---------------------------------------------------------------------------
# load_scenario()'s own default for solar_build_case is "nirp_trajectory" (a
# cumulative solar cap). Runners that never passed the keyword inherited it.
# These two flags let such a runner be run at another solar_build_case WITHOUT
# changing its default behaviour: with neither flag given, nothing here alters
# a single argument, so every stored Phase B result stays reproducible.
#   --solar-build-case  absent -> no keyword is passed (the inherited default).
#   --results-suffix    absent -> the runner's own RESULTS_DIR. Given -> a
#                       sibling directory <RESULTS_DIR><suffix> (so the capped
#                       results survive as the NIRP-trajectory policy arm).
# A non-default --solar-build-case REQUIRES a --results-suffix: it must never
# write into the directory that holds the capped results.

def add_solar_build_args(parser):
    parser.add_argument(
        "--solar-build-case", dest="solar_build_case", default=None,
        choices=sorted(solar_build_scenarios()),
        help="Solar build arm fed to load_scenario(solar_build_case=...). "
             "ABSENT = pass nothing (load_scenario's own default, "
             "nirp_trajectory). Requires --results-suffix when given.")
    parser.add_argument(
        "--results-suffix", dest="results_suffix", default="",
        help="Write to <RESULTS_DIR><suffix> (a sibling directory) instead of "
             "RESULTS_DIR, e.g. _uncapped.")


def solar_build_kwargs(args):
    """{} unless --solar-build-case was given; then {"solar_build_case": case}."""
    case = getattr(args, "solar_build_case", None)
    if case is None:
        return {}
    if not getattr(args, "results_suffix", ""):
        raise SystemExit(
            "ABORT: --solar-build-case requires --results-suffix, so the "
            "capped results in the default directory are never overwritten.")
    return {"solar_build_case": case}


# ---------------------------------------------------------------------------
# TASK 53 2.5 (G7-4): arm values reach every runner's output key.
# ---------------------------------------------------------------------------
# Before: 22 of 28 runners wrote fixed-name outputs (a second invocation at a different arm
# silently overwrote the first), `us_inflation` was in no key although it deflates every capital
# charge, and runner 29 had no key at all. Now every runner's output location carries a hash of the
# FULL arm set it was run at. Runners that already keyed their files extend their tag with
# complete_arm_tag(); runners that wrote fixed names go through resolve_results_dir(), which now
# always returns <results dir>/arms_<hash8>/ (with ARMS.json holding the full tag, and an INDEX.md
# line in the parent). This moves every hash and path; it is free only because the epoch's re-run
# moves them anyway.

def _scenario_default_arms():
    """The load_scenario() defaults of arms that are NOT command-line flags, so a change of default changes the key."""
    import inspect
    from src.scenarios import load_scenario
    params = inspect.signature(load_scenario).parameters
    return {k: params[k].default for k in ("capital_window_case", "storage_capex_case", "gas_vintage_dispatch")}


def full_arm_dict(args, extra=None):
    """Every arm a runner was invoked at: standard arms, load_scenario defaults of the non-flag arms, and the runner's own."""
    kw = dict(standard_arms_kwargs(args))
    kw.update(_scenario_default_arms())
    for k in ("capital_case", "solar_build_case", "capital_window_case", "storage_capex_case"):
        v = getattr(args, k, None)
        if v is not None:
            kw[k] = v
    # TASK 74: gas_vintage_dispatch is on by default but forced off under exempt deliverability (load_scenario),
    # so the key states the EFFECTIVE value unless the runner passes its own through `extra`.
    if kw.get("new_gas_deliverability", "constrained") == "exempt":
        kw["gas_vintage_dispatch"] = False
    kw.update(extra or {})
    return kw


def complete_arm_tag(tag, args, in_tag=(), sep="__", extra=None):
    """`tag` plus every arm in full_arm_dict(args) whose kwarg name is not in `in_tag` (the arms the runner's own tag already encodes)."""
    kw = full_arm_dict(args, extra)
    return tag + "".join(f"{sep}{k}={v}" for k, v in sorted(kw.items()) if k not in set(in_tag))


def arm_key(args, extra=None):
    """(full tag, hash8) over every arm: the output key of a runner that wrote fixed names."""
    from src.output_naming import hash8_of
    tag = "__".join(f"{k}={v}" for k, v in sorted(full_arm_dict(args, extra).items()))
    return tag, hash8_of(tag)


def resolve_results_dir(results_dir, args, extra=None, keyed=True):
    """
    The runner's results directory for THIS arm set: <base>/arms_<hash8>/, where <base> is
    `results_dir` (or the sibling <name><suffix> under --results-suffix, with DESIGN.txt). The
    directory is created, holds ARMS.json (the full arm tag and the invocation), and <base>/INDEX.md
    records `hash8 -> tag`. The path is printed (`Results directory: ...`) so a driver can parse it.
    keyed=False: for a runner that already hashes the full arm set into its own file/directory names
    (via complete_arm_tag()); only the --results-suffix sibling is resolved, as before TASK 53.
    """
    import json
    import sys
    from pathlib import Path
    from src.output_naming import append_index
    base = Path(results_dir)
    suffix = getattr(args, "results_suffix", "")
    if suffix:
        base = base.parent / (base.name + suffix)
        base.mkdir(parents=True, exist_ok=True)
        (base / "DESIGN.txt").write_text(
            f"solar_build_case: {getattr(args, 'solar_build_case', None) or 'default (load_scenario inherits nirp_trajectory)'}\n"
            f"runner invocation: {' '.join(sys.argv)}\n")
    if not keyed:
        # the runner keys its own files with complete_arm_tag(); only the --results-suffix sibling is resolved here
        return base
    tag, h8 = arm_key(args, extra)
    out = base / f"arms_{h8}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ARMS.json").write_text(json.dumps({"arm_tag": tag, "hash8": h8, "invocation": sys.argv}, indent=2))
    append_index(base, h8, tag)
    print(f"Results directory: {out}")
    return out
