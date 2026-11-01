"""
09_run_rel1_feasibility.py  —  REL-1
=====================================

EPS AXIS: see docs/eps_axis_rationale.md for why this project's REL
scripts legitimately sweep different eps ranges, and why this one uses
reliability_levels_log()'s default (adequacy-style, 5-20% unserved).

RESEARCH QUESTION (REL-1)
--------------------------
Under a fixed capital envelope and solar build rate, does the feasibility
of minimum reliability standards depend primarily on gas deliverability
regime rather than on the level of solar investment?

WHAT THIS ANSWERS
-----------------
The question has a specific logical structure. It is not asking which
scenario achieves the best reliability — it is asking what DETERMINES
whether a given reliability standard is achievable at all.

The experiment holds capital and solar rate fixed, then asks:
  (a) Does the feasibility threshold shift across gas regimes?
      (gas regime = primary determinant hypothesis)
  (b) Does the feasibility threshold shift across solar build rates?
      (solar investment = alternative determinant hypothesis)

If (a) is large and (b) is small, the primary determinant is gas regime.
If both are large, both matter and the claim must be qualified.
If (b) is large and (a) is small, solar investment is the driver —
  this would falsify the primary hypothesis and is a valid finding.

EXPERIMENT DESIGN
-----------------
Fixed:
  capital_case = "moderate"  
  This is the baseline public capital assumption used throughout.
  It constrains solar deployment without being so tight that the
  system is trivially infeasible under all gas regimes.

Varied (gas — primary dimension):
  baseline, upside, downside, shock_recovery

Varied (solar deployment capability — control dimension):
  deployment_unconstrained  — no cap; maximum solar flexibility
  nirp_trajectory           — NIRP 2024 cumulative capacity (61.9 GW by 2045)

Reliability epsilon grid:
  [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]
  — VoLL-only baseline also recorded (eps=None)

Total solves: 4 gas × 2 solar × 7 points = 56 LP solves

WHY UNCONSTRAINED vs NIRP TRAJECTORY
-------------------------------------
The comparison must span conditions where solar deployment COULD plausibly
matter. The previous arms (500 vs 2000 MW/yr flat) were two unsourced
numbers; both were also wrong in shape, permitting more early build than
NIRP judges achievable and less late build.

The replacement spans a SOURCED national deployment trajectory against no
constraint at all. NIRP's cumulative path binds this model from 2027 to 2037
(shortfall peaking at 4,505 MW in 2032), so the arms are genuinely distinct.
If the feasibility threshold is unchanged across them, the claim that gas
regime rather than solar deployment is the primary determinant is supported
against a real deployment constraint rather than an invented one.

CAVEAT: under nirp_trajectory the model cannot serve demand through the 2030s
and the shortfall is priced at VoLL. Read FEASIBILITY and UNSERVED ENERGY from
that arm, not NPV, until the genset backstop lands (plan 2.6).

OUTPUTS
--------
TASK 03 (docs/task_log_03_rel1_mode.md): output path is now mode-aware,
results/rel1/<reliability_mode>/, so a run at "total" can never overwrite
a run at "annual" or vice versa.

  results/rel1/<mode>/feasibility_matrix.csv    — all (gas, solar, eps) rows
  results/rel1/<mode>/feasibility_threshold.csv — summary: threshold per (gas, solar)
  results/rel1/<mode>/shortfall_by_year.csv     — NEW: one row per (gas, solar,
                                                   capital_case, eps, year)
  results/rel1/<mode>/rel1_summary.json         — headline finding + interpretation

RELIABILITY MODE
-----------------
--reliability-mode is REQUIRED, no default (project's standard required-
key discipline). "total" reproduces this runner's historical behaviour
(a single horizon-wide shortfall budget); "annual" applies the shortfall
bound year by year instead. See docs/task_log_01_omnibus_inventory.md
Part 2 and docs/task_log_03_rel1_mode.md for why the per-year series
(shortfall_by_year.csv) is what makes the "total" vs "annual" comparison
interpretable rather than an unexplained null.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ
from src.optimize_experiments import (
    run_rel1_feasibility_matrix,
    compute_feasibility_threshold,
)
from src.cli_arms import add_standard_arms, standard_arms_kwargs

# ============================================================
# CONFIG
# ============================================================

# TASK 03 (docs/task_log_03_rel1_mode.md): RESULTS_DIR is now mode-aware
# (results/rel1/<reliability_mode>/) and so can only be resolved once
# --reliability-mode has been parsed -- moved into main(), below. No
# longer a module-level constant.

CANONICAL_VOLL  = "voll_mid"
CAPITAL_CASE    = "moderate"

GAS_CASES        = ["baseline", "upside", "downside", "shock_recovery"]
SOLAR_BUILD_CASES = ["deployment_unconstrained", "nirp_trajectory"]

# Eps grid — ordered from loose to tight.
# run_rel1_feasibility_matrix will use reliability_levels_log() by default,
# but explicit here for transparency and reproducibility.
#
# TASK 23, Part B.1 (docs/task_log_23b_rel1_arm_grid.md): extended with
# the same tail TASK 13 gave 10_run_rel2_marginal_cost.py's EPS_LEVELS
# (0.02, 0.01, 0.005, 0.002, 0.001), so REL-1 and REL-2 now share an
# identical grid -- not a new range invented here. The original six
# levels left 6 of this grid's 8 (gas_case x solar_build_case) cells
# thresholding at the 0.05 floor under "annual" mode (all 8 under
# "total"), confirmed against results/rel1/{total,annual}/
# feasibility_threshold.csv (Zeus's own real, already-solved item-14
# run, not touched by this task). Ordering convention PRESERVED:
# loosest-first, unchanged -- the five new, tighter points are APPENDED
# after the existing six, not inserted or reordered. Confirmed (see the
# report), not assumed: neither run_rel1_feasibility_matrix() (each eps
# solved independently, no carried state across the loop) nor
# compute_feasibility_threshold() (.min()/groupby, order-independent)
# has any order-dependent computation.
EPS_LEVELS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


# ============================================================
# MAIN
# ============================================================
CANONICAL_GENSET = "genset_central"


def parse_args():
    # task_runner_revival, Part 2A: this runner had NO argparse at all
    # before this edit -- its sole run_rel1_feasibility_matrix() call
    # omitted the standard arm set outright. Built fresh, add_standard_
    # arms(parser), following 05_run_fin3_capital_sweep.py's pattern --
    # excluding gas_deliverability_case (this runner's own swept subject,
    # GAS_CASES above) and new_gas_deliverability (run_rel1_feasibility_
    # matrix does not accept it at all, confirmed via inspect.signature).
    parser = argparse.ArgumentParser(
        description="09_run_rel1_feasibility: REL-1 reliability feasibility matrix"
    )
    add_standard_arms(parser, exclude=["gas_deliverability_case", "new_gas_deliverability"])
    # TASK 03 (docs/task_log_03_rel1_mode.md): standalone flag, NOT added
    # to src/cli_arms.py's shared _arm_specs() registry -- that registry
    # is inherited by every OTHER runner calling add_standard_arms(parser)
    # with no exclude= for this key, and this task's own scope forbids
    # touching reliability_mode in any runner but this one. Required, no
    # default, matching the project's standard-arm discipline: an
    # unstated mode would silently pick one of two materially different
    # reliability formulations (docs/task_log_01_omnibus_inventory.md
    # Part 2).
    parser.add_argument(
        "--reliability-mode", dest="reliability_mode",
        choices=["total", "annual"], required=True,
        help="Reliability constraint formulation fed to run_rel1_"
             "feasibility_matrix(reliability_mode=...). 'total' is the "
             "pre-TASK-03 behaviour (a single horizon-wide shortfall "
             "budget, the deprecated grid_service_mode alias this "
             "runner has always used). 'annual' applies the shortfall "
             "bound year by year instead, matching every other live "
             "reliability runner (REL-2, REL-3). No default: the two "
             "modes are materially different formulations, not a "
             "performance knob.",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    standard_kwargs = standard_arms_kwargs(args)

    # TASK 03 (docs/task_log_03_rel1_mode.md), Section A, point 4: output
    # path is mode-aware so a "total" run can never overwrite an "annual"
    # one or vice versa.
    #
    # TASK 23, Part B.2 (docs/task_log_23b_rel1_arm_grid.md): the
    # gas_availability_case arm now nests INSIDE reliability_mode --
    # results/rel1/<mode>/<gas_availability_case>/, not the reverse.
    # Chosen, not defaulted: `reliability_mode` is the more fundamental
    # axis (it selects which of two materially different LP constraint
    # formulations is solved, per TASK 03's own framing -- "not a
    # performance knob"), already established as this runner's outer
    # partition since TASK 03, and item 14's own four-invocation
    # structure (two modes, each then run under two arms) reads the
    # same way: decide the formulation first, then the data scenario
    # within it. Nesting the arm inside preserves TASK 03's existing
    # structure unchanged for any reader who already treats
    # results/rel1/<mode>/ as a stable prefix, rather than inserting a
    # new segment ahead of it.
    results_dir = ROOT / "results" / "rel1" / args.reliability_mode / args.gas_availability_case
    results_dir.mkdir(parents=True, exist_ok=True)

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    print(f"\nREL-1: Reliability feasibility matrix")
    print(f"  capital_case:      {CAPITAL_CASE}")
    print(f"  gas_cases:         {GAS_CASES}")
    print(f"  solar_build_cases: {SOLAR_BUILD_CASES}")
    print(f"  eps_levels:        {EPS_LEVELS}")
    print(f"  reliability_mode:  {args.reliability_mode}")
    print(f"  total solves:      {len(GAS_CASES) * len(SOLAR_BUILD_CASES) * (1 + len(EPS_LEVELS))}")

    # TASK 03: run_rel1_feasibility_matrix() now returns (all_rows,
    # year_rows) instead of a single list -- year_rows is new (Section B).
    rows, year_rows = run_rel1_feasibility_matrix(
        econ=econ,
        gas_cases=GAS_CASES,
        solar_build_cases=SOLAR_BUILD_CASES,
        builder="sliced",  # A5 addendum, Part 3: was hardwired to the annual builder
        reliability_mode=args.reliability_mode,
        capital_case=CAPITAL_CASE,
        eps_levels=EPS_LEVELS,
        **standard_kwargs,
    )

    # ── Save full matrix ───────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    # TASK 23, Part B.2: gas_availability_case as a column in every REL-1
    # output file, stamped on here exactly as TASK 13/22 did for REL-2/
    # REL-3 -- one invocation of this script always runs under a single
    # gas_availability_case value (the CLI arm), so a constant column is
    # exact.
    df["gas_availability_case"] = args.gas_availability_case
    matrix_path = results_dir / "feasibility_matrix.csv"
    df.to_csv(matrix_path, index=False)
    print(f"\nSaved: {matrix_path}")

    # ── Save per-year shortfall series (TASK 03, Section B) ────────────────
    year_df = pd.DataFrame(year_rows)
    year_path = results_dir / "shortfall_by_year.csv"
    year_df.to_csv(year_path, index=False)
    n_cells_with_years = year_df[["gas_case", "solar_build_case", "eps"]].drop_duplicates().shape[0] if len(year_df) else 0
    n_cells_total = df[["gas_case", "solar_build_case", "eps"]].drop_duplicates().shape[0]
    print(f"Saved: {year_path} ({len(year_df)} year rows, "
          f"{n_cells_with_years}/{n_cells_total} (cell, eps) combinations contributed rows)")

    # ── Compute and save feasibility thresholds ────────────────────────────
    threshold_df = compute_feasibility_threshold(rows, args.gas_availability_case)
    threshold_path = results_dir / "feasibility_threshold.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(f"Saved: {threshold_path}")

    # TASK 23, Part B.3 (docs/task_log_23b_rel1_arm_grid.md): make
    # floor-binding visible at the summary level, matching TASK 22's
    # Part C for REL-3. "Cell" = one (gas_case, solar_build_case) row in
    # threshold_df.
    n_threshold_cells_total = len(threshold_df)
    n_threshold_cells_floor_bound = int(threshold_df["threshold_is_floor_bound"].sum())
    threshold_degeneracy = {
        "n_cells_total": n_threshold_cells_total,
        "n_cells_floor_bound": n_threshold_cells_floor_bound,
        "cells_floor_bound": [
            {"gas_case": r["gas_case"], "solar_build_case": r["solar_build_case"]}
            for _, r in threshold_df[threshold_df["threshold_is_floor_bound"]].iterrows()
        ],
        # True when the MAJORITY of cells are floor-bound -- so the
        # PRIMARY FINDING below (and any reader of it in isolation)
        # cannot mistake "small measured range" for "small true effect,"
        # when it may instead mean "the grid never tested tight enough."
        "majority_cells_floor_bound": n_threshold_cells_floor_bound > n_threshold_cells_total / 2,
    }
    eps_floor_for_print = min(EPS_LEVELS)

    # ── Build summary ──────────────────────────────────────────────────────
    # Variation across gas regimes (within each solar arm)
    # Variation across solar arms (within each gas regime)
    # If gas variation >> solar variation => gas is primary determinant

    threshold_df["threshold_numeric"] = threshold_df["feasibility_threshold_eps"].astype(float)

    gas_variation = {}
    for solar_case in SOLAR_BUILD_CASES:
        sub = threshold_df[threshold_df["solar_build_case"] == solar_case]
        thresholds = sub["threshold_numeric"].dropna()
        gas_variation[solar_case] = {
            "min_threshold": float(thresholds.min()) if len(thresholds) else None,
            "max_threshold": float(thresholds.max()) if len(thresholds) else None,
            "range":         float(thresholds.max() - thresholds.min()) if len(thresholds) else None,
            "n_infeasible":  int(sub["fully_infeasible"].sum()),
        }

    solar_variation = {}
    for gas_case in GAS_CASES:
        sub = threshold_df[threshold_df["gas_case"] == gas_case]
        solar_variation[gas_case] = {
            "unconstrained_threshold": (
                sub[sub["solar_build_case"] == "deployment_unconstrained"]["feasibility_threshold_eps"].values[0]
                if len(sub[sub["solar_build_case"] == "deployment_unconstrained"]) else None
            ),
            "nirp_threshold": (
                sub[sub["solar_build_case"] == "nirp_trajectory"]["feasibility_threshold_eps"].values[0]
                if len(sub[sub["solar_build_case"] == "nirp_trajectory"]) else None
            ),
            "solar_arm_difference": (
                float(
                    sub[sub["solar_build_case"] == "deployment_unconstrained"]["threshold_numeric"].values[0]
                    - sub[sub["solar_build_case"] == "nirp_trajectory"]["threshold_numeric"].values[0]
                )
                if (
                    len(sub[sub["solar_build_case"] == "deployment_unconstrained"]) and
                    len(sub[sub["solar_build_case"] == "nirp_trajectory"]) and
                    sub["threshold_numeric"].notna().all()
                )
                else None
            ),
        }

    # Primary finding: does gas regime shift threshold more than solar level?
    max_gas_range   = max(
        (v["range"] for v in gas_variation.values() if v["range"] is not None),
        default=None
    )
    max_solar_diff  = max(
        (abs(v["solar_arm_difference"]) for v in solar_variation.values()
         if v["solar_arm_difference"] is not None),
        default=None
    )

    # TASK 23, Part C.1 (docs/task_log_23c_rel2_display.md): checked, and
    # 09_ has the same problem now that Part B.1 extended EPS_LEVELS to
    # include 0.01/0.005/0.002/0.001 -- these ranges/diffs are drawn from
    # the same grid REL-2's binding-region table collapsed at .2f.
    # Widened to .3f throughout this block, matching REL-2's own fix.
    if max_gas_range is not None and max_solar_diff is not None:
        if max_gas_range > max_solar_diff:
            primary_determinant = "gas_regime"
            interpretation = (
                f"Gas regime shifts the feasibility threshold by up to {max_gas_range:.3f} "
                f"(in unserved fraction units), while varying solar deployment capability shifts it by "
                f"at most {max_solar_diff:.3f}. Gas deliverability is the primary determinant."
            )
        elif max_solar_diff > max_gas_range:
            primary_determinant = "solar_investment"
            interpretation = (
                f"Solar deployment capability shifts the feasibility threshold by up to {max_solar_diff:.3f}, "
                f"exceeding the gas regime effect of {max_gas_range:.3f}. "
                f"Solar deployment capability is the primary determinant — the hypothesis is not supported."
            )
        else:
            primary_determinant = "indeterminate"
            interpretation = "Gas regime and solar deployment capability shift the threshold by equal amounts."
    else:
        primary_determinant = "indeterminate"
        interpretation = "Insufficient feasible solutions to determine primary driver."

    # TASK 23, Part B.3: the PRIMARY FINDING must say plainly when its
    # own determination rests on floor-bound cells -- max_gas_range and
    # max_solar_diff are both RANGES over feasibility_threshold_eps, and
    # a floor-bound threshold on either side of either range is a lower
    # bound on the true effect (clipped by the floor), never an upper
    # one. This does not change primary_determinant or interpretation's
    # own wording above -- it appends a caveat, stated once, rather than
    # leaving the majority-floor-bound case silently indistinguishable
    # from a confirmed small effect.
    if threshold_degeneracy["majority_cells_floor_bound"]:
        interpretation += (
            f" CAVEAT: {n_threshold_cells_floor_bound} of "
            f"{n_threshold_cells_total} cells never left the tightest "
            f"tested eps ({eps_floor_for_print:.3f}) -- a majority. The "
            f"measured ranges above are lower bounds on the true gas-"
            f"regime and solar-arm effects, not confirmed magnitudes; a "
            f"tighter grid could reveal a larger shift on either axis."
        )

    summary = {
        "experiment": {
            "capital_case":      CAPITAL_CASE,
            "gas_cases":         GAS_CASES,
            "solar_build_cases": SOLAR_BUILD_CASES,
            "eps_levels":        EPS_LEVELS,
            "voll":              CANONICAL_VOLL,
            # TASK 03, Section A, point 3.
            "reliability_mode":  args.reliability_mode,
            # TASK 23, Part B.2: the arm that produced this file, so a
            # standalone rel1_summary.json is never ambiguous about which
            # availability assumption it came from.
            "gas_availability_case": args.gas_availability_case,
        },
        "primary_determinant":   primary_determinant,
        "interpretation":        interpretation,
        "max_gas_range_eps":     max_gas_range,
        "max_solar_diff_eps":    max_solar_diff,
        "gas_variation_by_solar_arm":  gas_variation,
        "solar_variation_by_gas_case": solar_variation,
        # TASK 23, Part B.3: visible at the summary level, naming
        # mirroring TASK 22's "frontier_degeneracy" block for REL-3.
        "threshold_degeneracy": threshold_degeneracy,
        "threshold_table": threshold_df[[
            "gas_case", "solar_build_case",
            "feasibility_threshold_eps", "fully_infeasible",
            "threshold_is_floor_bound",
            "npv_cost_at_threshold",
        ]].to_dict(orient="records"),
    }

    json_path = results_dir / "rel1_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n=== REL-1 FEASIBILITY THRESHOLD TABLE ===")
    print(f"  Fixed: capital_case={CAPITAL_CASE}")
    print()
    print(f"  {'Gas case':<18} {'Solar build':<14} {'Threshold eps':>14} {'Infeasible?':>12}")
    print("  " + "-" * 62)
    for _, row in threshold_df.sort_values(
        ["solar_build_case", "gas_case"]
    ).iterrows():
        # TASK 23, Part C.1: .2f collapses 0.01/0.005/0.002/0.001 (Part
        # B.1's own extension) into two ambiguous labels; widened to .3f.
        thresh = (
            f"{row['feasibility_threshold_eps']:.3f}"
            if pd.notna(row["feasibility_threshold_eps"])
            else "NONE"
        )
        infeas = "YES" if row["fully_infeasible"] else "no"
        floor = " <-FLOOR" if row["threshold_is_floor_bound"] else ""
        print(f"  {row['gas_case']:<18} {row['solar_build_case']:<14} {thresh:>14} {infeas:>12}{floor}")

    print()
    print(f"=== PRIMARY FINDING ===")
    print(f"  Primary determinant: {primary_determinant.upper()}")
    print(f"  {interpretation}")
    print()
    print(f"  Max threshold range across gas regimes: {max_gas_range}")
    print(f"  Max threshold shift across solar arms:  {max_solar_diff}")

    # TASK 23, Part B.3: degeneracy visible at the console too, same
    # discipline as TASK 22's Part C for REL-3.
    print()
    print("=== THRESHOLD DEGENERACY (TASK 23, Part B.3) ===")
    print(f"  Cells thresholding at the eps floor ({eps_floor_for_print:.3f}): "
          f"{n_threshold_cells_floor_bound}/{n_threshold_cells_total}")
    print(f"  Majority floor-bound: {threshold_degeneracy['majority_cells_floor_bound']}")
    if threshold_degeneracy["majority_cells_floor_bound"]:
        print("  WARNING: most cells never found a confirmed feasibility "
              "boundary -- the PRIMARY FINDING above measures a lower "
              "bound on the gas-regime and solar-arm effects, not their "
              "true magnitude.")


if __name__ == "__main__":
    main()
