"""
11_run_rel3_financing_frontier.py  —  REL-3
============================================

EPS AXIS: see docs/eps_axis_rationale.md for why this project's REL
scripts legitimately sweep different eps ranges, and why this one
widens to [0.45..0.20] rather than reliability_levels_log()'s default.

RESEARCH QUESTION (REL-3)
--------------------------
Does EaaS financing improve reliability outcomes (measured by the
feasibility frontier) relative to public-capital-only financing under
the same gas constraint? Is this still important under NDC3.0?

WHY REL-3 IS MORE IMPORTANT UNDER NDC3.0
-----------------------------------------
Under NDC3.0, two constraints interact simultaneously:

  (A) grid_service_obligation: sum(selfgen + unserved) <= eps × total_demand
      NOTE (2.6): this bounds GRID SERVICE SHORTFALL, not adequacy. A household
      on its own genset was not grid-served but no capacity was short. With a
      backstop, unserved alone is ~0 by construction (D11), so a constraint
      written on unserved would never bind.
  (B) emissions_constraint[t]: emissions[t] <= cap_t  (annual NDC cap)

Under public-capital-only financing, the public budget caps solar deployment.
If the budget is exhausted before sufficient solar is built to both (a) meet
the reliability target AND (b) substitute for gas emissions, the system is
infeasible — both constraints cannot be satisfied simultaneously.

EaaS adds private solar deployment beyond the public budget. This additional
solar serves BOTH objectives at once:
  - Reduces unserved energy (improving reliability)
  - Substitutes for gas generation (reducing emissions)

The NDC3.0 conditional scenario (47% reduction) is
the most stringent test: concessional finance makes more EaaS bankable,
but the emissions cap is tighter. The interaction is non-trivial.

WHAT "FEASIBILITY FRONTIER IMPROVEMENT" MEANS
----------------------------------------------
The feasibility frontier is the set of (eps, cost) pairs where the
system CAN achieve reliability standard eps at total cost C.

EaaS improves the frontier if:
  threshold_eps(EaaS) < threshold_eps(public_only)
  i.e. EaaS achieves tighter standards (smaller eps still feasible)

  CAVEAT (2.6, D17), UPDATED (task_defect_closure, Part 4): at the default
  tariff the EaaS arm HAD no binding capital constraint -- it built until
  demand was served and returned the same answer to every input, making
  frontier_shift_eps measure constrained-vs-unconstrained rather than a
  financing effect. That was written before tariff_case existed as a
  required, sourced arm (docs/tariff_arm.md); it is no longer an assumption
  this runner must carry silently. D17 is now DETECTABLE PER RUN: every
  EaaS row records eaas_bankability_n_binding_years /
  eaas_bankability_n_evaluable_years (reusing
  03_run_ndc_eaas.py's bankability_slack_per_mw_by_year approach --
  "binds" means slack < 1 USD/MW, the same threshold
  docs/findings_provisional_pre_sweep.md's F-29 already used for this
  exact quantity), and frontier_shift.csv's eaas_not_a_financing_effect
  column is True whenever the EaaS arm's bankability constraint binds in
  zero of the evaluable years AT THE THRESHOLD eps used to compute that
  row's shift -- i.e. whenever this specific caveat applies to this
  specific row. Rows where it is False are measuring an actual financing
  effect; rows where it is True are constrained-vs-unconstrained
  artefacts and must be read with that caveat, not silently trusted. See
  FIN-2 (04_run_fin2_tariff_sweep.py) for the tariff sweep that first
  made tariff_case a real, swept arm.

The frontier_shift_eps metric quantifies this:
  frontier_shift = threshold(public_only) - threshold(eaas)
  Positive: EaaS achieves tighter reliability
  Zero:     EaaS provides no feasibility benefit at this gas/policy combo
  Null:     Cannot compute — one or both arms fully infeasible

Special case (EaaS necessary, not just beneficial):
  If public_only is fully infeasible (cannot achieve ANY eps) but EaaS
  is feasible, EaaS is NECESSARY for any reliability standard to be met.
  This is the strongest possible finding.

EXPERIMENT STRUCTURE
--------------------
  gas_cases:     4 regimes (baseline, upside, downside, shock_recovery)
  policy_arms:   3 levels (no_policy, ndc3_unconditional, ndc3_conditional)
  financing_arms: 2 (public_only, eaas)
  eps_levels:    [0.45, 0.40, 0.35, 0.30, 0.25, 0.20]

  Total solves: 4 × 3 × 2 × 7 = 168

FINANCING ARM PARAMETERS
------------------------
  public_only:
    financing_regime = "traditional"
    public_capital = "unconstrained" (no_policy), "tight" (unconditional),
      "moderate" (conditional) -- matches 02_run_ndc_caps.py (TASK 15,
      Part C.1, docs/task_log_15c_documentation_defects.md: this
      docstring previously named a "capital_case" field, which does not
      exist on POLICY_CONFIGS below, and claimed "moderate"/"expansion";
      the actual field is "public_capital"/"eaas_capital" and the actual
      values are "unconstrained"/"tight"/"moderate", confirmed by reading
      POLICY_CONFIGS directly. Comment corrected to match the code; the
      code itself is unchanged)

  eaas:
    financing_regime = "eaas"
    tariff_case = tariff_legacy_95m (95_000_000 USD/TWh; docs/tariff_arm.md)
    required_margin: NOT set here. dscr_case-derived (load_scenario()), the
      same single cushion as the public arm (TASK 29, register L34: the
      pre-F1 1.10/1.05 split is RETIRED as a double-count; the former
      "eaas_margin" literals overrode it on this arm only)
    eaas_capital matches public_capital per policy (confirmed in
      POLICY_CONFIGS: identical value on both keys, every row)

OUTPUTS
--------
  results/rel3/frontier_all.csv          — all (gas, policy, financing, eps) rows
  results/rel3/frontier_threshold.csv    — threshold per combination
  results/rel3/frontier_shift.csv        — shift metric per (gas, policy)
  results/rel3/rel3_summary.json         — structured findings
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.cli_arms import add_solar_build_args, solar_build_kwargs, resolve_results_dir  # TASK 46
from src.io import load_econ
from src.scenarios import tariff_scenarios
from src.optimize_experiments import (
    run_rel3_financing_frontier,
    compute_frontier_shift,
    reliability_results_to_df,
)
from src.cli_arms import add_standard_arms, standard_arms_kwargs

# ============================================================
# CONFIG
# ============================================================

# TASK 22, Part B (docs/task_log_22_rel3_grid.md): RESULTS_DIR is now
# gas_availability_case-aware (results/rel3/<gas_availability_case>/,
# the same convention TASK 13 applied to 10_run_rel2_marginal_cost.py --
# results/rel2/<gas_availability_case>/ -- itself matching TASK 03's
# results/rel1/<reliability_mode>/) and so can only be resolved once
# --gas-availability-case has been parsed -- moved into main(), below.
# No longer a module-level constant.

CAP_PATH       = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"
# docs/tariff_arm.md, Steps 1-3: was a bare `95_000_000` literal threaded
# into run_rel3_financing_frontier via each POLICY_CONFIG's "eaas_tariff".
# This runner is non-functional independent of this task -- the backend's
# load_scenario() call omits ~15 arms required since phases 2.6-2.13 and
# raises before returning. The bare literal is replaced by the named
# retained-for-attribution legacy arm, threaded via
# load_scenario(tariff_case=...) rather than a post-call mutation.
CANONICAL_TARIFF_CASE = "tariff_legacy_95m"
CANONICAL_GENSET = "genset_central"

GAS_CASES = ["baseline", "upside", "downside", "shock_recovery"]

# TASK 22, Part A (docs/task_log_22_rel3_grid.md): extended downward,
# matching TASK 13's own treatment of 10_run_rel2_marginal_cost.py's
# EPS_LEVELS exactly -- do not invent a second convention for the same
# problem. The original six levels (0.45..0.20) left 18 of this grid's
# 20 (gas_case x policy_label) cells thresholding at the 0.20 floor
# (confirmed empirically, not asserted -- see the report): the system
# remained feasible all the way to the tightest tested point, so the
# TRUE threshold is unknown, not found. The five new points
# (0.15, 0.10, 0.08, 0.06, 0.05) are exactly REL-2's own original grid
# floor region (10_'s pre-TASK-13 EPS_LEVELS, minus the 0.20 it shares
# with this grid already) -- "the region REL-1 and REL-2 found
# informative," per this task's own framing, not a new range invented
# here. Ordering convention PRESERVED: loosest-first, unchanged: the
# five new, tighter points are APPENDED after the existing six, not
# inserted or reordered. Confirmed (see the report) that
# compute_frontier_shift() has no order-dependent computation of its
# own (it uses .min()/.max()/groupby, all order-independent) -- the
# convention is preserved here for consistency with 10_'s own grid and
# this file's own console/JSON display order, not because a computation
# would silently break otherwise.
EPS_LEVELS = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05]

# Policy configurations — one dict per policy arm
# Parameters set the financing arm details for that policy context
POLICY_CONFIGS = [
    {
        "policy_label":   "no_policy",
        "ndc_scenario":   None,              # no emissions cap
        "public_capital": "unconstrained",
        "eaas_capital":   "unconstrained",
        "tariff_case":    CANONICAL_TARIFF_CASE,
    },
    {
        "policy_label":   "ndc2_unconditional",
        "ndc_scenario":   "ndc2_unconditional",
        "public_capital": "tight",
        "eaas_capital":   "tight",
        "tariff_case":    CANONICAL_TARIFF_CASE,
    },
    {
        "policy_label":   "ndc2_conditional",
        "ndc_scenario":   "ndc2_conditional",
        "public_capital": "moderate",
        "eaas_capital":   "moderate",
        "tariff_case":    CANONICAL_TARIFF_CASE,
    },
    {
        "policy_label":   "ndc3_unconditional",
        "ndc_scenario":   "ndc3_unconditional",
        "public_capital": "tight",        # matches 02_run_ndc_caps.py
        "eaas_capital":   "tight",
        "tariff_case":    CANONICAL_TARIFF_CASE,
    },
    {
        "policy_label":   "ndc3_conditional",
        "ndc_scenario":   "ndc3_conditional",
        "public_capital": "moderate",       # matches 02_run_ndc_caps.py
        "eaas_capital":   "moderate",
        "tariff_case":    CANONICAL_TARIFF_CASE,
    },
]


# ============================================================
# MAIN
# ============================================================

def parse_args():
    # task_runner_revival, Part 2A: this runner had NO argparse at all
    # before this edit -- its sole run_rel3_financing_frontier() call
    # omitted the standard arm set outright. Built fresh, add_standard_
    # arms(parser), following 05_run_fin3_capital_sweep.py's pattern --
    # excluding gas_deliverability_case and new_gas_deliverability
    # (run_rel3_financing_frontier does not accept either, confirmed via
    # inspect.signature) and excluding tariff_case (this function takes NO
    # top-level tariff_case parameter at all -- each POLICY_CONFIG above
    # already carries its own "tariff_case": CANONICAL_TARIFF_CASE entry,
    # a per-arm value this revival does not change, per "change no
    # constant, no computation").
    parser = argparse.ArgumentParser(
        description="11_run_rel3_financing_frontier: REL-3 EaaS-vs-public-only reliability frontier"
    )
    add_standard_arms(parser, exclude=["gas_deliverability_case", "new_gas_deliverability", "tariff_case"])
    add_solar_build_args(parser)  # TASK 46: opt-in; absent = inherited default
    return parser.parse_args()


def main():

    args = parse_args()
    standard_kwargs = standard_arms_kwargs(args)
    # TASK 46: opt-in design flag; absent = no keyword = inherited default.
    standard_kwargs.update(solar_build_kwargs(args))

    # TASK 22, Part B: results/rel3/<gas_availability_case>/, same
    # convention as TASK 13's results/rel2/<gas_availability_case>/ --
    # one arm's run can never overwrite the other's output.
    results_dir = resolve_results_dir(ROOT / "results" / "rel3", args) / args.gas_availability_case
    results_dir.mkdir(parents=True, exist_ok=True)

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    n_solves = len(GAS_CASES) * len(POLICY_CONFIGS) * 2 * (1 + len(EPS_LEVELS))
    print(f"\nREL-3: EaaS vs public-only financing frontier")
    print(f"  gas_cases:    {GAS_CASES}")
    print(f"  policy_arms:  {[p['policy_label'] for p in POLICY_CONFIGS]}")
    print(f"  eps_levels:   {EPS_LEVELS}")
    print(f"  total solves: {n_solves}")

    # ── Run all combinations ───────────────────────────────────────────────
    all_rows = run_rel3_financing_frontier(
        econ=econ,
        gas_cases=GAS_CASES,
        policy_configs=POLICY_CONFIGS,
        builder="sliced",  # A5 addendum, Part 3: was hardwired to the annual builder
        eps_levels=EPS_LEVELS,
        cap_path=str(CAP_PATH),
        **standard_kwargs,
    )

    df = reliability_results_to_df(all_rows)

    # Tag financing_label and policy_label from the new row fields
    # (reliability_results_to_df may not carry these through — add them)
    for col in ["financing_label", "policy_label"]:
        if col not in df.columns:
            df[col] = [r.get(col, "") for r in all_rows]

    # TASK 22, Part B: gas_availability_case as a column in every REL-3
    # output file, stamped on here exactly as TASK 13 did for REL-2's
    # cost_curve_all.csv -- one invocation of this script always runs
    # under a single gas_availability_case value (the CLI arm), so a
    # constant column is exact. Not threaded through run_reliability_
    # sweep/_make_row (src/optimize_experiments.py, shared with REL-2),
    # keeping this change confined to REL-3's own functions/output.
    df["gas_availability_case"] = args.gas_availability_case

    df.to_csv(results_dir / "frontier_all.csv", index=False)
    print(f"\nSaved: {results_dir / 'frontier_all.csv'}")

    # ── Compute thresholds and shifts ──────────────────────────────────────
    threshold_df, shift_df = compute_frontier_shift(df, args.gas_availability_case)

    threshold_df.to_csv(results_dir / "frontier_threshold.csv", index=False)
    shift_df.to_csv(results_dir / "frontier_shift.csv", index=False)
    print(f"Saved: {results_dir / 'frontier_threshold.csv'}")
    print(f"Saved: {results_dir / 'frontier_shift.csv'}")

    # ── Build summary JSON ─────────────────────────────────────────────────
    summary = {
        "experiment": {
            "gas_cases":     GAS_CASES,
            "policy_arms":   [p["policy_label"] for p in POLICY_CONFIGS],
            "eps_levels":    EPS_LEVELS,
            "tariff_case": CANONICAL_TARIFF_CASE,
            "canonical_tariff_usd_per_twh":
                tariff_scenarios()[CANONICAL_TARIFF_CASE],
            "voll":          CANONICAL_VOLL,
            # TASK 22, Part B: the arm that produced this file, so a
            # standalone rel3_summary.json is never ambiguous about which
            # availability assumption it came from.
            "gas_availability_case": args.gas_availability_case,
        },
        "findings_by_policy": {},
        "ndc_relevance": {},
    }

    for pol in [p["policy_label"] for p in POLICY_CONFIGS]:
        pol_shift = shift_df[shift_df["policy_label"] == pol]

        # Cases where EaaS is strictly better (positive shift)
        better = pol_shift[pol_shift["eaas_strictly_better"] == True]
        necessary = pol_shift[pol_shift["eaas_necessary"] == True]

        summary["findings_by_policy"][pol] = {
            "n_gas_cases_where_eaas_strictly_better": int(len(better)),
            "n_gas_cases_where_eaas_necessary":       int(len(necessary)),
            "max_frontier_shift_eps": (
                float(pol_shift["frontier_shift_eps"].max())
                if pol_shift["frontier_shift_eps"].notna().any() else None
            ),
            "gas_cases_eaas_necessary":  necessary["gas_case"].tolist(),
            "gas_cases_eaas_better":     better["gas_case"].tolist(),
            "shift_by_gas_case": pol_shift[
                ["gas_case", "pub_threshold_eps", "eaas_threshold_eps",
                 "frontier_shift_eps", "frontier_classification",
                 "eaas_strictly_better", "eaas_necessary",
                 "eaas_sole_route_eps_range",
                 "eaas_bankability_n_binding_years_at_threshold",
                 "eaas_not_a_financing_effect", "cell_is_floor_bound",
                 "interpretation"]
            ].to_dict(orient="records"),
        }

    # TASK 22, Part C (docs/task_log_22_rel3_grid.md): make degeneracy
    # visible at the summary level, not just per-row. "Cell" = one
    # (gas_case, policy_label) combination in shift_df -- the same
    # granularity Part A's own "18 of 20 cells" finding used. A result
    # here is read as a substantive negative (no improvement / not
    # necessary) only if cell_is_floor_bound is False for that cell;
    # this block exists so that distinction is never lost by only
    # reading frontier_shift_eps/eaas_necessary in isolation.
    n_cells_floor_bound = int(shift_df["cell_is_floor_bound"].sum())
    n_cells_total = len(shift_df)
    summary["frontier_degeneracy"] = {
        "n_cells_total": n_cells_total,
        "n_cells_floor_bound": n_cells_floor_bound,
        "cells_floor_bound": [
            {"gas_case": r["gas_case"], "policy_label": r["policy_label"]}
            for _, r in shift_df[shift_df["cell_is_floor_bound"]].iterrows()
        ],
        # True when the MAJORITY of cells are floor-bound -- so a reader
        # of frontier_shift_eps/eaas_necessary alone (without checking
        # this block) cannot mistake "most cells show no difference"
        # for "EaaS makes no difference," when it may instead mean "the
        # grid never tested tight enough to find out."
        "majority_cells_floor_bound": n_cells_floor_bound > n_cells_total / 2,
    }

    # NDC relevance: does EaaS improvement grow under tighter NDC caps?
    no_pol  = summary["findings_by_policy"].get("no_policy", {})
    unc_pol = summary["findings_by_policy"].get("ndc3_unconditional", {})
    con_pol = summary["findings_by_policy"].get("ndc3_conditional", {})

    max_shift_no_pol = no_pol.get("max_frontier_shift_eps")
    max_shift_unc    = unc_pol.get("max_frontier_shift_eps")
    max_shift_con    = con_pol.get("max_frontier_shift_eps")

    summary["ndc_relevance"] = {
        "max_shift_no_policy":        max_shift_no_pol,
        "max_shift_ndc_unconditional": max_shift_unc,
        "max_shift_ndc_conditional":   max_shift_con,
        # TASK 23, Part D.2 (docs/task_log_23d_rel3_prose.md): formula
        # UNCHANGED -- confirmed nothing else in the repository reads
        # this field (grep -rn "eaas_improvement_grows_with_ndc_ambition"
        # --include="*.py" --include="*.md" . -- only this computation
        # and its own console print, below). Kept exactly as it was:
        # compares the CONDITIONAL arm against no_policy only, never
        # examining the unconditional arm at all -- a narrower test than
        # "grows with ambition" sounds like, which is exactly why the
        # interpretation below no longer relies on this boolean alone.
        "eaas_improvement_grows_with_ndc_ambition": (
            (max_shift_con is not None and max_shift_no_pol is not None
             and max_shift_con > max_shift_no_pol)
            if (max_shift_con is not None and max_shift_no_pol is not None)
            else None
        ),
    }

    # TASK 23, Part D.1/D.2: the old text here asserted "EaaS may be
    # sufficient without NDC pressure" whenever max_shift_no_policy was
    # not strictly greater than max_shift_ndc_conditional -- including
    # the real case where max_shift_no_policy is exactly 0.0 (EaaS adds
    # NOTHING without policy pressure, the opposite of "sufficient"),
    # and described the three-point pattern as a flat "does not
    # increase" even when the true shape is an interior maximum
    # (0.0 / 0.10 / 0.0 at this gas regime, confirmed against
    # results/rel3/gas_avail_nameplate/rel3_summary.json, Zeus's own
    # real run, untouched by this task) -- correct about the two
    # ENDPOINTS the boolean above compares, silent about the peak
    # between them. Rewritten to describe the actual shape across all
    # three points. No computation changed: max_shift_no_pol/unc/con
    # are the same three numbers already computed above; this only
    # reads them more fully than the boolean alone does.
    if max_shift_no_pol is not None and max_shift_unc is not None and max_shift_con is not None:
        if max_shift_unc > max_shift_no_pol and max_shift_unc > max_shift_con:
            ndc_interpretation = (
                f"EaaS's frontier benefit has an interior maximum, not a "
                f"monotonic relationship with NDC ambition: {max_shift_no_pol:.3f} "
                f"with no policy pressure, {max_shift_unc:.3f} under the "
                f"unconditional arm, {max_shift_con:.3f} under the conditional "
                f"arm. EaaS is worth nothing when financing faces no policy "
                f"pressure and nothing again once the emissions cap is tight "
                f"enough that financing cannot relieve it either way -- its "
                f"value is concentrated at moderate ambition, not at either "
                f"extreme."
            )
        elif max_shift_unc < max_shift_no_pol and max_shift_unc < max_shift_con:
            ndc_interpretation = (
                f"EaaS's frontier benefit has an interior MINIMUM across NDC "
                f"ambition: {max_shift_no_pol:.3f} (no_policy), {max_shift_unc:.3f} "
                f"(unconditional), {max_shift_con:.3f} (conditional) -- lower at "
                f"moderate ambition than at either extreme."
            )
        elif max_shift_no_pol == max_shift_unc == max_shift_con:
            ndc_interpretation = (
                f"EaaS's frontier benefit does not vary with NDC ambition at "
                f"this gas regime: {max_shift_no_pol:.3f} across no_policy, "
                f"unconditional and conditional arms alike."
            )
        elif max_shift_no_pol <= max_shift_unc <= max_shift_con:
            ndc_interpretation = (
                f"EaaS's frontier benefit rises monotonically with NDC "
                f"ambition: {max_shift_no_pol:.3f} (no_policy), "
                f"{max_shift_unc:.3f} (unconditional), {max_shift_con:.3f} "
                f"(conditional)."
            )
        elif max_shift_no_pol >= max_shift_unc >= max_shift_con:
            ndc_interpretation = (
                f"EaaS's frontier benefit falls monotonically with NDC "
                f"ambition: {max_shift_no_pol:.3f} (no_policy), "
                f"{max_shift_unc:.3f} (unconditional), {max_shift_con:.3f} "
                f"(conditional)."
            )
        else:
            ndc_interpretation = (
                f"EaaS's frontier benefit does not move monotonically with "
                f"NDC ambition: {max_shift_no_pol:.3f} (no_policy), "
                f"{max_shift_unc:.3f} (unconditional), {max_shift_con:.3f} "
                f"(conditional)."
            )
    else:
        ndc_interpretation = "Insufficient data to assess NDC relevance."

    summary["ndc_relevance"]["interpretation"] = ndc_interpretation

    with open(results_dir / "rel3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {results_dir / 'rel3_summary.json'}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== REL-3 FRONTIER SHIFT TABLE ===")
    print("  (frontier_shift > 0 means EaaS achieves tighter reliability)")
    print()

    for pol in [p["policy_label"] for p in POLICY_CONFIGS]:
        print(f"\n  Policy: {pol}")
        print(f"  {'Gas regime':<18} {'Pub thresh':>11} {'EaaS thresh':>12} "
              f"{'Shift eps':>10} {'EaaS better':>12} {'Pub infeas.':>11}")
        print("  " + "-" * 76)

        pol_rows = shift_df[shift_df["policy_label"] == pol].sort_values("gas_case")
        for _, row in pol_rows.iterrows():
            pt  = f"{row['pub_threshold_eps']:.2f}"  if pd.notna(row["pub_threshold_eps"])  else "infeas"
            et  = f"{row['eaas_threshold_eps']:.2f}" if pd.notna(row["eaas_threshold_eps"]) else "infeas"
            sh  = f"{row['frontier_shift_eps']:+.2f}" if pd.notna(row["frontier_shift_eps"]) else "—"
            # task_defect_closure, D24: both columns now read off the same
            # frontier_classification, so a necessary row reads YES/YES,
            # never YES/no -- see compute_frontier_shift().
            bet = "YES" if row["eaas_strictly_better"] else "no"
            nec = "YES" if row["eaas_necessary"] else "no"
            print(f"  {row['gas_case']:<18} {pt:>11} {et:>12} {sh:>10} {bet:>12} {nec:>11}")
            if row["frontier_classification"] == "eaas_worse":
                print(f"    ANOMALY: {row['interpretation']}")
            # TASK 22, Part C: a floor-bound cell's "no"/"—" above is not
            # confirmed -- the grid never tested tight enough on at least
            # one side to know if that reads differently past the floor.
            if row["cell_is_floor_bound"]:
                print(
                    f"    FLOOR-BOUND: pub={row['pub_threshold_is_floor_bound']} "
                    f"eaas={row['eaas_threshold_is_floor_bound']} -- shift/necessity "
                    f"here is untested-past-the-floor, not a confirmed negative."
                )
            # task_defect_closure, D17: printed warning, same spirit as
            # this project's LEAKAGE FLAG prints elsewhere -- do not
            # suppress frontier_shift_eps, label it. eaas_not_a_financing_
            # effect is None (not True/False) when there was no EaaS
            # threshold row to check at all (e.g. eaas fully infeasible);
            # only warn on an affirmative True.
            if row.get("eaas_not_a_financing_effect") is True:
                nb = row.get("eaas_bankability_n_binding_years_at_threshold")
                print(
                    f"    D17 WARNING: EaaS bankability constraint binds in "
                    f"{nb} evaluable year(s) at the threshold eps -- "
                    f"frontier_shift_eps here is constrained-vs-"
                    f"unconstrained, NOT a financing effect."
                )

    print()
    print("=== NDC RELEVANCE ===")
    ndc = summary["ndc_relevance"]
    print(f"  Max shift — no_policy:        {ndc['max_shift_no_policy']}")
    print(f"  Max shift — ndc_unconditional: {ndc['max_shift_ndc_unconditional']}")
    print(f"  Max shift — ndc_conditional:   {ndc['max_shift_ndc_conditional']}")
    print(f"  EaaS grows with NDC ambition: {ndc['eaas_improvement_grows_with_ndc_ambition']}")
    print(f"  {ndc['interpretation']}")

    # TASK 22, Part C: degeneracy visible at the console, not just in
    # rel3_summary.json -- the same discipline as the FLOOR-BOUND warning
    # printed per-row above, restated once here as the headline number.
    deg = summary["frontier_degeneracy"]
    print()
    print("=== FRONTIER DEGENERACY (TASK 22, Part C) ===")
    print(f"  Cells thresholding at the eps floor: "
          f"{deg['n_cells_floor_bound']}/{deg['n_cells_total']}")
    print(f"  Majority floor-bound: {deg['majority_cells_floor_bound']}")
    if deg["majority_cells_floor_bound"]:
        print("  WARNING: most cells never found a confirmed feasibility "
              "boundary -- most of the shift/necessity table above "
              "reflects where testing stopped, not where EaaS and "
              "public-only financing actually diverge or do not.")


if __name__ == "__main__":
    main()
