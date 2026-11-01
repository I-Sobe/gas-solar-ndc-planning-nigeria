"""
10_run_rel2_marginal_cost.py  —  REL-2
=======================================

EPS AXIS: see docs/eps_axis_rationale.md for why this project's REL
scripts legitimately sweep different eps ranges, and why this one uses
reliability_levels_log()'s default (adequacy-style, 5-20% unserved).

RESEARCH QUESTION (REL-2)
--------------------------
What is the marginal cost of tightening the reliability standard, and
how does this cost curve change across gas deliverability regimes?

WHAT THE DUAL MEANS
-------------------
TASK 02 correction (docs/task_log_02_dual_fix.md): this section
previously stated the constraint in "total" (horizon-wide) mode and
omitted `selfgen[t]`, both stale against HEAD -- confirmed by direct
read, not reused from this section's own prior wording. REL-2 actually
calls `run_reliability_sweep` with `grid_service_mode="annual"`
(src/optimize_experiments.py), i.e. ONE constraint per calendar year,
not a single horizon-wide budget, and the obligation bounds grid
SHORTFALL (selfgen + unserved), not unserved alone -- a 2.6 correction
this section predated. The constraint, per year t:

    selfgen[t] + unserved[t] <= eps * demand[t]

where eps is the maximum allowed shortfall fraction (e.g. 0.10 = 10%).

In a minimisation LP with a <= constraint, the dual is non-positive.
The sign is negated to get the shadow price as a positive number. The
dual is a PRESENT-VALUE marginal (the objective is entirely NPV-
denominated), and it is a per-YEAR quantity under "annual" mode, so a
single scalar dual does not exist here at all -- `extract_reliability_
dual()` now returns it as several explicit statistics rather than one
collapsed number (docs/task_log_02_dual_fix.md), read from the columns
below:

    dual_pv_usd_per_twh_binding_mean / _all_mean
        present-value dual, averaged over years where the constraint
        ACTUALLY binds (own primal slack < 1e-4 TWh) / over every year
        with a recorded dual regardless of bindingness. These are two
        different numbers -- a non-binding year returns dual 0 under LP
        complementary slackness, which drags the "_all_mean" toward zero
        relative to "_binding_mean" in proportion to how many years are
        slack. The economically meaningful one is "_binding_mean".
    dual_pv_usd_per_twh_binding_median / _min / _max
        same binding-years set, further summary statistics.
    dual_yr_usd_per_twh_binding_mean / _all_mean
        the same two means, each year's present-value dual first divided
        by that year's own m.DF[t] -- a same-year-dollars marginal
        instead of a present-value one. Because DF[2054] ~ 0.107 at
        r_social=0.08, a year-2054 scarcity value of the same REAL
        magnitude as a year-2030 one entered the old collapsed mean
        roughly an order of magnitude smaller; the yr-basis columns are
        not subject to that distortion.
    binding_years_n / years_with_dual_n
        how many years the means above are actually drawn from -- an
        empty set is None, not 0.0, in every mean/median/min/max column.

This is the marginal system cost reduction per additional TWh of
shortfall ALLOWED (i.e., per unit of RHS relaxation) in a binding year.

The NORMALISED forms (one per pv/yr x binding/all combination above,
same `_pv_`/`_yr_` naming) are more interpretable for the thesis:

    dual_pv_usd_per_pct_pt_binding_mean = dual_pv_usd_per_twh_binding_mean * total_demand_twh / 100
    dual_pv_usd_per_pct_pt_all_mean     = dual_pv_usd_per_twh_all_mean     * total_demand_twh / 100
    dual_yr_usd_per_pct_pt_binding_mean = dual_yr_usd_per_twh_binding_mean * total_demand_twh / 100
    dual_yr_usd_per_pct_pt_all_mean     = dual_yr_usd_per_twh_all_mean     * total_demand_twh / 100

This is the marginal cost of improving reliability by 1 percentage point
(e.g. from 90% to 91% served). Units: USD NPV per pp. The normalisation
constant (total_demand_twh / 100) and its source
(src/optimize_experiments.py, inside `run_reliability_sweep`) are
unchanged by TASK 02 -- only the dual being normalised changed shape.

`dual_usd_per_twh` and `dual_usd_per_pct_pt` are RETIRED as of TASK 02
(docs/task_log_02_dual_fix.md) -- no alias is kept. Code or notes
written against either old name will raise `KeyError`
(`src/optimize_experiments.py`) or read `None` from a `.get()`
(this file, below), not silently return a diluted number.

WHAT THE COST CURVE LOOKS LIKE
-------------------------------
As eps tightens from 0.20 -> 0.05:
  - At loose eps levels, the constraint may be non-binding (slack).
    The system achieves better reliability than required WITHOUT the
    constraint forcing it. Dual = 0, slack > 0.
  - Once eps crosses below the system's natural reliability level,
    the constraint becomes binding. Dual > 0, slack ≈ 0.
  - As eps tightens further, the dual rises — each additional pp of
    reliability requires increasingly expensive measures (more solar,
    storage, or accepting lower utilisation of gas plants).

The REL-2 thesis claim is that this curve SHIFTS LEFT across gas regimes:
  - Under downside gas, the constraint becomes binding at a LOOSER eps
    (higher unserved fraction) because the system's natural reliability
    is lower. The curve starts rising earlier.
  - Under upside gas, the constraint stays slack until eps is very tight.
  - The curve's STEEPNESS also differs: downside may produce a steeper
    curve (each pp is more expensive when gas is scarce).

The "left shift" and steepness difference together quantify how much
gas deliverability regime drives the cost of reliability.

WHAT IS DIFFERENT FROM REL-1
------------------------------
REL-1 asked: does gas regime determine FEASIBILITY (can the standard
be met at all)?  -> binary: feasible/infeasible at each eps.

REL-2 asks: what is the MARGINAL COST of meeting the standard?
-> continuous: how steeply does cost rise as eps tightens?

REL-1 uses the primal (feasibility) as its signal.
REL-2 uses the dual (shadow price) as its signal.
Both are needed: REL-1 establishes that gas regime determines
WHETHER reliability is achievable; REL-2 establishes at what COST.

BINDING DETECTION
-----------------
A dual of zero can mean either:
  (a) constraint is slack (system achieves the target without being forced)
  (b) degenerate LP solution (constraint binding but dual is zero)

To distinguish, primal slack is computed directly:
    primal_slack = eps * total_demand - sum(unserved)

If primal_slack < 1e-4 TWh: constraint is binding (dual is meaningful)
If primal_slack >= 1e-4 TWh: constraint is slack (dual of zero is uninformative)

Only binding-region duals are used to construct the cost curve.

OUTPUTS
--------
  results/rel2/cost_curve_all.csv      — all rows (all gas regimes, all eps)
  results/rel2/cost_curve_binding.csv  — binding region only (dual meaningful)
  results/rel2/curve_summary.csv       — onset eps + steepness per gas regime
  results/rel2/rel2_summary.json       — left-shift metric + interpretation
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
from src.optimize_experiments import (
    run_rel2_marginal_cost_curves,
    reliability_results_to_df,
)
from src.cli_arms import add_standard_arms, standard_arms_kwargs

# ============================================================
# CONFIG
# ============================================================

# TASK 13, Part B (docs/task_log_13_rel2_grid.md): RESULTS_DIR is now
# gas_availability_case-aware (results/rel2/<gas_availability_case>/,
# same convention TASK 03 used for reliability_mode -- results/rel1/
# <reliability_mode>/) and so can only be resolved once
# --gas-availability-case has been parsed -- moved into main(), below.
# No longer a module-level constant.

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"
CAPITAL_CASE   = "moderate"    # Fixed — same as REL-1 for comparability

GAS_CASES = ["baseline", "upside", "downside", "shock_recovery"]

# Eps grid: 0.20 -> 0.001 (loose to tight). TASK 13: extended from the
# original six levels (0.20..0.05) with five tighter points
# (0.02, 0.01, 0.005, 0.002, 0.001) -- TASK 09/10 found baseline/upside's
# realised ceilings sit at 0 (gas_avail_nameplate) or 0.0025-0.011
# (gas_avail_observed), both well below the original grid's 0.05 floor,
# so the original grid could never find an onset for three of the four
# regimes regardless of availability arm. Loosest-first ordering is
# PRESERVED (unchanged convention) -- compute_curve_summary's onset
# detection (`onset_eps = binding["eps"].max()`) and the console/CSV
# sort order both depend on it; only new, tighter entries were appended
# after the existing six, not inserted or reordered.
EPS_LEVELS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def compute_curve_summary(df, gas_availability_case):
    """
    For each gas_case, compute:
      onset_eps:  the loosest eps at which the constraint is binding
                  (where the cost curve starts rising)
      onset_reliability: 1 - onset_eps
      steepness:  slope of dual_pv_usd_per_pct_pt_binding_mean vs
                  reliability in binding region (positive = steeper curve
                  = more expensive reliability)
      max_dual:   peak dual_pv_usd_per_pct_pt_binding_mean in binding region

    TASK 02 (docs/task_log_02_dual_fix.md): reads `dual_pv_usd_per_pct_
    pt_binding_mean`, not the retired `dual_usd_per_pct_pt` -- the
    present-value dual averaged over years that actually bind, which is
    what "max dual in binding region" and "steepness" were always meant
    to describe, not the old all-years-diluted mean.

    TASK 13, Part B: `gas_availability_case` is stamped onto every row so
    a summary row is never ambiguous about which availability assumption
    produced it (same discipline TASK 03 applied to `reliability_mode`
    in `feasibility_matrix.csv`). Not read from `df` itself -- one
    invocation of this script always runs under a single
    `gas_availability_case` value (the CLI arm), so it is passed in
    directly rather than re-derived per row.

    Returns a DataFrame with one row per gas_case.
    """
    rows = []
    for gas_case, grp in df.groupby("gas_case"):
        feasible = grp[grp["status"] == "feasible"].copy()
        # TASK 15, Part A.2 (docs/task_log_15a_binding_criterion.md):
        # switched from the retired "binding" (a horizon-aggregate
        # comparison that could not see a single binding year out of
        # thirty) to "binding_any_year" (binding_years_n >= 1, sourced
        # from extract_reliability_dual's own per-year result). Nothing
        # else in this function changes.
        binding  = feasible[feasible["binding_any_year"] == True].copy()

        # Onset: loosest eps where binding (largest eps value in binding region)
        onset_eps = float(binding["eps"].max()) if len(binding) > 0 else None
        onset_rel = 1 - onset_eps if onset_eps is not None else None

        # Max dual in binding region
        max_dual = (
            float(binding["dual_pv_usd_per_pct_pt_binding_mean"].max())
            if len(binding) > 0 else None
        )

        # Steepness: slope of (dual vs reliability) in binding region
        # Using simple linear regression across the binding eps levels
        steepness = None
        if len(binding) >= 2:
            x = binding["eps"].values           # x = eps (decreasing = tighter)
            y = binding["dual_pv_usd_per_pct_pt_binding_mean"].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 2:
                # As eps decreases (tighter standard), dual should increase
                # Steepness = |d(dual)/d(eps)| — larger = steeper curve
                slope = np.polyfit(x[valid], y[valid], 1)[0]
                steepness = float(-slope)  # negate: we want rate of increase as eps falls

        rows.append({
            "gas_case":          gas_case,
            "gas_availability_case": gas_availability_case,
            "onset_eps":         onset_eps,
            "onset_reliability": onset_rel,
            "max_dual_usd_per_pct_pt": max_dual,
            "steepness_usd_per_pct_pt_per_eps": steepness,
            "n_binding_eps":     len(binding),
            "n_infeasible_eps":  int((grp["status"] == "infeasible").sum()),
        })

    return pd.DataFrame(rows)


def compute_left_shift(summary_df):
    """
    Quantify the left shift of the cost curve across gas regimes.

    Left shift = how much earlier (at a looser eps) does the cost curve
    start rising compared to the best-case (upside) gas regime?

    onset_eps(upside) - onset_eps(gas_case):
      Positive = gas_case curve starts rising earlier (left-shifted)
      Zero     = same onset as upside
      Negative = gas_case curve starts rising later (right-shifted, cheaper reliability)
    """
    upside_row = summary_df[summary_df["gas_case"] == "upside"]
    upside_onset = (
        float(upside_row["onset_eps"].values[0])
        if len(upside_row) and upside_row["onset_eps"].values[0] is not None
        and not pd.isna(upside_row["onset_eps"].values[0])
        else None
    )

    rows = []
    for _, r in summary_df.iterrows():
        onset = r["onset_eps"]
        left_shift = (
            float(upside_onset - onset)
            if (upside_onset is not None and onset is not None)
            else None
        )
        rows.append({
            "gas_case":       r["gas_case"],
            "onset_eps":      onset,
            "left_shift_eps": left_shift,
            # Positive left_shift means constraint binds earlier under this regime
            "curve_is_left_shifted": (
                left_shift > 1e-3 if left_shift is not None else None
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================



def parse_args():
    # task_runner_revival, Part 2A: this runner had NO argparse at all
    # before this edit -- its sole run_rel2_marginal_cost_curves() call
    # omitted the standard arm set outright. Built fresh, add_standard_
    # arms(parser), following 05_run_fin3_capital_sweep.py's pattern --
    # excluding gas_deliverability_case (this runner's own swept subject,
    # GAS_CASES above) and new_gas_deliverability (run_rel2_marginal_cost_
    # curves does not accept it at all, confirmed via inspect.signature).
    parser = argparse.ArgumentParser(
        description="10_run_rel2_marginal_cost: REL-2 marginal cost of reliability"
    )
    add_standard_arms(parser, exclude=["gas_deliverability_case", "new_gas_deliverability"])
    # TASK 27 Part B: rebuild every post-solve output (cost_curve_binding,
    # curve_summary, rel2_summary.json) from the STORED cost_curve_all.csv,
    # with no solve and without rewriting cost_curve_all.csv itself. Added
    # because rel2_summary.json for gas_avail_nameplate predates TASK 23
    # Part C's prose fix and re-running its 48 solves was the alternative.
    parser.add_argument(
        "--regenerate-from-stored", action="store_true",
        help="Skip all solves; read results/rel2/<arm>/cost_curve_all.csv "
             "and regenerate the derived outputs from it.",
    )
    add_solar_build_args(parser)  # TASK 46: opt-in; absent = inherited default
    return parser.parse_args()


def main():

    args = parse_args()
    standard_kwargs = standard_arms_kwargs(args)
    # TASK 46: opt-in design flag; absent = no keyword = inherited default.
    standard_kwargs.update(solar_build_kwargs(args))

    # TASK 13, Part B: results/rel2/<gas_availability_case>/, same
    # convention as TASK 03's results/rel1/<reliability_mode>/ -- one
    # arm's run can never overwrite the other's output.
    results_dir = resolve_results_dir(ROOT / "results" / "rel2", args) / args.gas_availability_case
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.regenerate_from_stored:
        stored = results_dir / "cost_curve_all.csv"
        print(f"\nREL-2: REGENERATING from stored {stored} -- no solve")
        df = pd.read_csv(stored)
    else:
        econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

        print(f"\nREL-2: Marginal cost of reliability across gas regimes")
        print(f"  capital_case:  {CAPITAL_CASE}")
        print(f"  gas_cases:     {GAS_CASES}")
        print(f"  eps_levels:    {EPS_LEVELS}")
        print(f"  total solves:  {len(GAS_CASES) * (1 + len(EPS_LEVELS))}")

        # ── Run all sweeps ─────────────────────────────────────────────────────
        all_rows = run_rel2_marginal_cost_curves(
            econ=econ,
            gas_cases=GAS_CASES,
            builder="sliced",  # A5 addendum, Part 3: was hardwired to the annual builder
            capital_case=CAPITAL_CASE,
            eps_levels=EPS_LEVELS,
            **standard_kwargs,
        )

        df = reliability_results_to_df(all_rows)

    # TASK 13, Part B.1: gas_availability_case as a column in every REL-2
    # output file. Stamped on here rather than threaded through
    # run_reliability_sweep/_make_row (src/optimize_experiments.py,
    # shared with REL-3) -- one invocation of this script always runs
    # under a single arm value, so a constant column is exact, and this
    # keeps the change confined to this file (11_run_rel3_financing_
    # frontier.py is out of scope for this task and is not touched).
    df["gas_availability_case"] = args.gas_availability_case

    # ── Save full results ──────────────────────────────────────────────────
    if not args.regenerate_from_stored:
        df.to_csv(results_dir / "cost_curve_all.csv", index=False)

    # Binding-region only (where dual is economically meaningful).
    # TASK 15, Part A.2/A.3: switched from "binding" (horizon-aggregate,
    # confirmed by TASK 13's regression to leave this file header-only
    # even at the one cell known to bind) to "binding_any_year" -- this
    # file is what A.3's regression checks is no longer empty at
    # downside/eps=0.20.
    binding_df = df[
        (df["binding_any_year"] == True) &
        (df["status"] == "feasible") &
        (df["eps"].notna())
    ].copy()
    binding_df.to_csv(results_dir / "cost_curve_binding.csv", index=False)

    print(f"\nSaved: {results_dir / 'cost_curve_all.csv'}")
    print(f"Saved: {results_dir / 'cost_curve_binding.csv'}")

    # ── Curve summary ──────────────────────────────────────────────────────
    summary_df = compute_curve_summary(df, args.gas_availability_case)
    summary_df.to_csv(results_dir / "curve_summary.csv", index=False)

    # ── Left-shift analysis ────────────────────────────────────────────────
    shift_df = compute_left_shift(summary_df)

    # ── Build summary JSON ─────────────────────────────────────────────────
    summary = {
        "experiment": {
            "capital_case":  CAPITAL_CASE,
            "gas_cases":     GAS_CASES,
            "eps_levels":    EPS_LEVELS,
            "voll":          CANONICAL_VOLL,
            # TASK 13, Part B.3: the arm that produced this file, so a
            # standalone rel2_summary.json is never ambiguous about which
            # availability assumption it came from.
            "gas_availability_case": args.gas_availability_case,
        },
        "curve_by_gas_case": {},
        "left_shift_analysis": {},
        "interpretation": None,
    }

    for _, row in summary_df.iterrows():
        gc = row["gas_case"]
        shift_row = shift_df[shift_df["gas_case"] == gc].iloc[0]
        summary["curve_by_gas_case"][gc] = {
            "onset_eps":                 row["onset_eps"],
            "onset_reliability":         row["onset_reliability"],
            "max_dual_usd_per_pct_pt":   row["max_dual_usd_per_pct_pt"],
            "steepness":                 row["steepness_usd_per_pct_pt_per_eps"],
            "n_binding_eps":             int(row["n_binding_eps"]),
            "n_infeasible_eps":          int(row["n_infeasible_eps"]),
        }
        summary["left_shift_analysis"][gc] = {
            "onset_eps":          row["onset_eps"],
            "left_shift_eps":     shift_row["left_shift_eps"],
            "curve_left_shifted": shift_row["curve_is_left_shifted"],
        }

    # TASK 13, Part C: make the dropped-regime problem visible instead of
    # silent. The left shift above is computed only over regimes with a
    # non-None onset_eps (unchanged -- this does not alter that
    # computation), but a reader of rel2_summary.json alone previously
    # had no way to tell how many regimes were actually compared versus
    # silently dropped. regimes_without_onset is read directly off
    # summary_df (the same frame onset_vals above is derived from), not
    # recomputed independently.
    # TASK 23, Part C.2 (docs/task_log_23c_rel2_display.md): FOUND A BUG
    # while fixing the NaN prose this part names, not just a display
    # issue -- the same None-vs-NaN defect TASK 22's own "D24" finding
    # named for compute_frontier_shift(). `summary["left_shift_analysis"]`
    # is built from summary_df (a DataFrame); a column mixing Python
    # `None` (no onset found) with real floats is coerced to float64 by
    # pandas, turning every missing onset into NaN, NOT None. The `is
    # None`/`is not None` checks below therefore NEVER caught a missing
    # onset whenever at least one other regime had a real one (every
    # real run) -- regimes_without_onset came back empty and
    # left_shift_is_degenerate came back False even when it was True,
    # confirmed directly against results/rel2/gas_avail_nameplate/
    # rel2_summary.json (Zeus's own real run, untouched by this task):
    # it reports n_regimes_with_onset=4/4, regimes_without_onset=[],
    # left_shift_is_degenerate=False, while left_shift_analysis itself
    # shows onset_eps=nan for BOTH baseline and upside -- 2 of 4, not 0.
    # pd.isna()/pd.notna() replace every is-None test below, exactly as
    # D24 did for compute_frontier_shift(); this is confined to this
    # script's own main(), not src/optimize_experiments.py -- compute_
    # curve_summary() and compute_left_shift() themselves are unchanged.
    regimes_without_onset = [
        gc for gc, d in summary["left_shift_analysis"].items()
        if pd.isna(d["onset_eps"])
    ]
    n_regimes_total = len(summary["left_shift_analysis"])
    n_regimes_with_onset = n_regimes_total - len(regimes_without_onset)
    summary["left_shift_degeneracy"] = {
        "n_regimes_with_onset": n_regimes_with_onset,
        "n_regimes_total": n_regimes_total,
        "regimes_without_onset": regimes_without_onset,
        # True when fewer than 2 regimes have an onset -- a "left shift"
        # computed over 0 or 1 regimes compares a regime to itself (or
        # to nothing) rather than to a genuine alternative.
        "left_shift_is_degenerate": n_regimes_with_onset < 2,
    }

    # Identify the regime with the most left-shifted curve (earliest onset)
    onset_vals = {
        gc: d["onset_eps"]
        for gc, d in summary["left_shift_analysis"].items()
        if pd.notna(d["onset_eps"])
    }
    # TASK 23, Part C.1 (docs/task_log_23c_rel2_display.md): .2f rounds
    # 0.01/0.005/0.002/0.001 (EPS_LEVELS' own tightest four levels, TASK
    # 13) into two ambiguous labels ("0.01" twice, "0.00" twice).
    # Widened to .3f, confirmed (see the report) to resolve every one of
    # this grid's eleven levels distinctly.
    if onset_vals:
        most_expensive = max(onset_vals, key=lambda k: onset_vals[k])
        least_expensive = min(onset_vals, key=lambda k: onset_vals[k])
        summary["interpretation"] = (
            f"Under {most_expensive} gas, the reliability cost curve starts "
            f"binding at eps={onset_vals[most_expensive]:.3f} (reliability="
            f"{1-onset_vals[most_expensive]:.3f}), meaning reliability becomes "
            f"costly at lower standards. Under {least_expensive} gas, the curve "
            f"only starts binding at eps={onset_vals[least_expensive]:.3f} "
            f"(reliability={1-onset_vals[least_expensive]:.3f}). "
            f"The left shift between these extremes is "
            f"{onset_vals[most_expensive] - onset_vals[least_expensive]:.3f} "
            f"in eps units."
        )
        # TASK 23, Part C.2: name the regimes excluded from this
        # comparison, when the comparison is only partial, rather than
        # silently comparing fewer regimes than the reader would assume.
        if regimes_without_onset:
            summary["interpretation"] += (
                f" No onset was found at any tested eps for: "
                f"{', '.join(sorted(regimes_without_onset))} -- excluded "
                f"from this comparison, not included as a zero or ignored "
                f"silently."
            )
    else:
        # TASK 23, Part C.2: the case this part names directly -- every
        # regime's onset is missing, so the old code fell through with
        # summary["interpretation"] left at its initial None, which TASK
        # 13's own console print (line ~507, unchanged) then formatted
        # into "eps=nan (reliability=nan) ... nan in eps units." Say
        # plainly what left_shift_is_degenerate (now correctly True
        # here) already means, naming every regime concerned.
        summary["interpretation"] = (
            f"No onset was found at any tested eps for any gas regime "
            f"({', '.join(sorted(summary['left_shift_analysis'].keys()))}) "
            f"-- the reliability constraint never became binding at the "
            f"tightest eps level tested. The left-shift comparison cannot "
            f"be computed; this is reported, not approximated."
        )

    with open(results_dir / "rel2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {results_dir / 'rel2_summary.json'}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== REL-2 COST CURVE SUMMARY ===")
    print(f"\n  {'Gas regime':<18} {'Onset eps':>10} {'Onset rel':>10} "
          f"{'Max dual (M$/pp)':>17} {'Steepness':>10} {'Left shift':>11}")
    print("  " + "-" * 80)

    for _, row in summary_df.sort_values("onset_eps", ascending=False).iterrows():
        gc    = row["gas_case"]
        onset = f"{row['onset_eps']:.3f}" if pd.notna(row["onset_eps"]) else "—"
        orel  = f"{row['onset_reliability']:.3f}" if pd.notna(row["onset_reliability"]) else "—"
        maxd  = (
            f"{row['max_dual_usd_per_pct_pt']/1e6:.1f}"
            if pd.notna(row["max_dual_usd_per_pct_pt"]) else "—"
        )
        steep = (
            f"{row['steepness_usd_per_pct_pt_per_eps']/1e6:.1f}"
            if pd.notna(row["steepness_usd_per_pct_pt_per_eps"]) else "—"
        )
        shift_row = shift_df[shift_df["gas_case"] == gc]
        ls = (
            f"{shift_row['left_shift_eps'].values[0]:.3f}"
            if len(shift_row) and pd.notna(shift_row["left_shift_eps"].values[0])
            else "—"
        )
        print(f"  {gc:<18} {onset:>10} {orel:>10} {maxd:>17} {steep:>10} {ls:>11}")

    print()
    if summary["interpretation"]:
        print("  " + summary["interpretation"])

    print("\n=== BINDING REGION DUALS ===")
    print("  (Only rows where constraint is binding — dual is economically meaningful)")
    print()
    print(f"  {'Gas regime':<18} {'eps':>7} {'Dual (M$/TWh)':>14} "
          f"{'Dual (M$/pp)':>13} {'Slack TWh':>10}")
    print("  " + "-" * 66)
    for _, row in binding_df.sort_values(
        ["gas_case", "eps"], ascending=[True, False]
    ).iterrows():
        # TASK 02 (docs/task_log_02_dual_fix.md): binding-mean, pv basis --
        # dual_usd_per_twh/dual_usd_per_pct_pt are retired.
        d_twh = (
            f"{row['dual_pv_usd_per_twh_binding_mean']/1e6:.2f}"
            if pd.notna(row["dual_pv_usd_per_twh_binding_mean"]) else "—"
        )
        d_pp = (
            f"{row['dual_pv_usd_per_pct_pt_binding_mean']/1e6:.2f}"
            if pd.notna(row["dual_pv_usd_per_pct_pt_binding_mean"]) else "—"
        )
        slack = (
            f"{row['primal_slack_twh']:.4f}"
            if pd.notna(row["primal_slack_twh"]) else "—"
        )
        # TASK 23, Part C.1: this is the table the task names directly --
        # .2f collapsed EPS_LEVELS' tightest four points (0.01/0.005/
        # 0.002/0.001) into two ambiguous labels ("0.01" twice, "0.00"
        # twice). Widened to .3f (column width 6->7 to match).
        print(f"  {row['gas_case']:<18} {row['eps']:>7.3f} "
              f"{d_twh:>14} {d_pp:>13} {slack:>10}")


if __name__ == "__main__":
    main()
