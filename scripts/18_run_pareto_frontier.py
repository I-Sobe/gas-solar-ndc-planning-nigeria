"""
18_run_pareto_frontier.py  —  Cost-Emissions Pareto Frontier
==============================================================

Traces the cost-emissions Pareto frontier for Nigeria's power sector
by sweeping the emissions cap from unconstrained (baseline) to very
tight, under both public-only and EaaS financing at tight capital.

TASK 26, Part B (docs/task_log_26b_pareto_anchors.md): the frontier was
originally reported with neither end located. The original grid,
linspace(1.0, 0.10, 25), has `cap_binding` true even at cap_fraction=1.0
and every arm feasible at 0.10, so it was a middle segment. Now, per
financing arm: (1) ONE uncapped "unconstrained anchor" solve, giving the
unconstrained optimum's emissions and the cap fraction at which a flat
cap stops binding (`loose_end_cap_fraction`); (2) the original 25-point
grid, unchanged; (3) tight-end extension points 0.08, 0.06, 0.05, 0.03,
0.02, 0.01, stopping at the arm's first infeasible point. Rows carry
`point_type` and `on_frontier`; pareto_endpoints.json carries per-arm
located/unlocated flags, boundary counts, and `endpoints_degenerate`,
with an overall `_endpoint_degeneracy` block.

This answers MIX-2: "How does the optimal mix change across the
cost-emissions Pareto frontier, and how does EaaS shift it?"

BUILDER: build_model_sliced (task_epsilon_constraint, Part 6B port)
-----------------------------------------------------------------------
This runner used the ANNUAL builder (`build_model`) through Parts 2-4
of this task. `methodology.md` §2 states the SLICED builder is the
reporting model, and every figure in `docs/findings_frozen.md` is
sliced. The gap was not cosmetic: the annual builder cannot represent
OCGT-to-CCGT conversion at all (`build_model(..., conversion_case=
"conv_central", ...)` raises `ValueError`), and conversion is the
second-largest lever in the model at **5.91 bn** (F-6), cutting capped
emissions 26-27% (F-8). An annual cost-emissions frontier therefore
traced the trade-off of a system with no conversion available at all —
compared beside a sliced central case, it was comparing two different
models. Ported per `docs/task_epsilon_constraint.md` Part 6's own
inventory: `build_model_sliced` has an identical call signature
(confirmed: both frontiers already call with 100% keyword arguments),
builds `m.emissions_constraint` and `m.public_budget_constraint` with
byte-for-byte identical structure to the annual builder (indexed by
year `t`, never by slice `(s,p)` — the slice summation happens inside
the underlying Expressions at build time), and needs no scenario-dict
key this runner does not already supply. `solve_model` and
`leakage_diagnostic` are builder-agnostic (confirmed: `01b_run_
baseline_sliced.py` uses both, unmodified, on a sliced model) and keep
importing from `src.optimize_model` unchanged.

The sweep still produces 25 points per financing arm (`N_POINTS`,
unchanged by this port — Part 2 changes what this runner solves with,
not what it sweeps), recording at each:
    - NPV system cost (total, real expenditure, VoLL)
    - Cumulative emissions
    - Solar deployed (MW)
    - Storage deployed (MWh)
    - Unserved energy (TWh)
    - System LCOE (USD/MWh)
    - the emissions-constraint dual (marginal abatement cost) and
      whether the cap binds (task_epsilon_constraint, Part 2)

BASELINE DENOMINATOR: --baseline-diagnostics, required, no default
-----------------------------------------------------------------------
The ε axis (`cap_fractions` below) is a fraction of this baseline
run's own annual emissions. This MUST be supplied explicitly, per
`--baseline-diagnostics <path>` (task_epsilon_constraint, Part 6B,
Part 3) — no glob, no fallback, no bare-path default. Two reasons:
(1) the bare `results/baseline/diagnostics.json` this runner
previously read by default is a stale, unlabelled 2026-08-23 file that
`01_run_baseline.py` does not refresh on a fresh run (it writes hash8-
tagged filenames instead); (2) `01_` is the ANNUAL runner and cannot
produce a `conv_central` baseline at all — the denominator must now
come from `01b_run_baseline_sliced.py`, which also writes hash8-tagged
files, never a bare `diagnostics.json`. An explicit required path
makes the denominator's provenance visible in the command that
produced the result — the same silent-default class this project has
now closed four times before this (the 65,000,000 USD/TWh tariff
default, `storage_duration_hours`, `om_case` hardcoded across six
library functions, `reserve_margin`); a runner silently selecting its
own input file by pattern would have been a fifth instance, not a fix.
The file given is validated (see `_validate_baseline_diagnostics`
below) to actually be a sliced, conversion-active baseline, and its
path and `arm_tag` are recorded in every output row — a frontier whose
ε axis depends on another run's output must carry that dependency in
its own results.

PRE-REQUISITES
--------------
    01b_run_baseline_sliced.py — sliced, conversion-active baseline
        diagnostics (its own hash8-tagged output file is what
        --baseline-diagnostics must point at). NOT 01_run_baseline.py
        (annual — cannot represent conversion; task_epsilon_
        constraint, Part 6B, Part 3).
    00_build_emissions_cap.py  — emissions cap trajectories

RUN SEQUENCE
------------
    python scripts/01b_run_baseline_sliced.py <required flags>
    python scripts/00_build_emissions_cap.py
    python scripts/18_run_pareto_frontier.py \\
        --baseline-diagnostics results/baseline_sliced/diagnostics__<hash8>.json \\
        <other required flags>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import load_scenario, asset_lifetime_sweep, solar_min_build_default, MODEL_END_YEAR
from src.optimize_model import solve_model, leakage_diagnostic
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics, annual_emissions_flat
from src.utils import json_safe
from src.cli_arms import add_standard_arms, standard_arms_kwargs, resolve_results_dir
from src.pareto_plot import (flag_dominance, draw_two_panel, dominated_points_table,
                             DOMINATED_NOTE, ARM_STYLES as arm_styles)

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

RESULTS_DIR = ROOT / "results" / "pareto_frontier"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

# task_epsilon_constraint, Part 6B, Part 3: the bare-path constant this
# used to be (ROOT / "results" / "baseline" / "diagnostics.json") is
# REMOVED, not merely unused -- --baseline-diagnostics (parse_args()
# below) is now the only source of this path, required with no
# default. See the module docstring's "BASELINE DENOMINATOR" section.

# Number of Pareto points
N_POINTS = 25

# TASK 26, Part B.2 (docs/task_log_26b_pareto_anchors.md): the original
# grid, np.linspace(1.0, 0.10, N_POINTS), stopped at 0.10 with every arm
# still feasible -- the tight end was never located. These points are
# appended BELOW 0.10, tightest last, mirroring the reliability suite's
# extended-grid convention (REL-1/2/3, TASK 13/22/23). Per arm they are
# solved in order and the arm STOPS at its first infeasible extension
# point rather than solving the whole tail. The N_POINTS grid itself is
# unchanged, so every pre-existing row is reproduced by the same inputs.
TIGHT_END_EXTENSION = [0.08, 0.06, 0.05, 0.03, 0.02, 0.01]

# Financing arms to compare
FINANCING_ARMS = [
    {
        "label":            "public_only_tight",
        "financing_regime": "public",
        "capital_case":     "tight",
    },
    {
        "label":            "eaas_tight",
        "financing_regime": "eaas",
        "capital_case":     "tight",
    },
    {
        "label":            "public_only_moderate",
        "financing_regime": "public",
        "capital_case":     "moderate",
    },
    {
        "label":            "eaas_moderate",
        "financing_regime": "eaas",
        "capital_case":     "moderate",
    },
]


# ── TASK 30, Part A: capital-envelope duals, recorded at every point ──────
# Two capital constraints exist in every model this runner builds:
#   public_budget_constraint            (NPV, horizon-aggregate; the FIN-3
#                                        constraint; value = capital_case)
#   f1_concessional_envelope_constraint (undiscounted commitment stock;
#                                        env_dares, the same USD in all arms)
# Each is ONE constraint over the whole horizon, not one per year, so the
# binding-year/evaluable-year convention (TASK 27 Part A) degenerates to
# n_binding in {0, 1} of n_evaluable in {0, 1}; there is no all-year mean to
# avoid and none is computed. The public-budget dual REUSES the FIN-3 path
# (extract_planning_diagnostics -> public_budget_shadow_usd_per_usd, sign
# convention -dual, clamped at 0); the f1 envelope dual is not extracted by
# that function, so it is read here directly with the same sign convention.
# Nothing here touches the model or the solve; any failure yields None
# fields, never an error row.
ENVELOPE_BINDING_TOL = 1e-6   # dual > 1e-6 == binding (TASK 27 Part A)


def _envelope_fields(m, diag, scenario, T):
    out = {
        "public_budget_npv_usd": None,
        "public_budget_shadow_usd_per_usd": None,
        "public_budget_utilisation": None,
        "public_budget_n_binding": None,
        "public_budget_n_evaluable": 0,
        "f1_envelope_usd": None,
        "f1_envelope_shadow_pv_per_undiscounted_usd": None,
        "f1_envelope_utilisation": None,
        "f1_envelope_n_binding": None,
        "f1_envelope_n_evaluable": 0,
        "envelope_dual_sign_anomaly": False,
    }
    try:
        out["public_budget_npv_usd"] = scenario.get("public_solar_budget_npv")
        sh = diag.get("public_budget_shadow_usd_per_usd")
        out["public_budget_shadow_usd_per_usd"] = sh
        out["public_budget_utilisation"] = diag.get("public_budget_utilisation")
        if sh is not None:
            out["public_budget_n_evaluable"] = 1
            out["public_budget_n_binding"] = int(sh > ENVELOPE_BINDING_TOL)
    except Exception:
        pass
    try:
        if hasattr(m, "f1_concessional_envelope_constraint"):
            cap_usd = float(scenario["concessional_envelope_usd"])
            out["f1_envelope_usd"] = cap_usd
            used = sum(
                float(pyo.value(m.solar_public_conc_vol[t]))
                + float(pyo.value(m.solar_eaas_conc_vol[t]))
                + float(pyo.value(m.storage_conc_vol[t]))
                for t in T
            )
            out["f1_envelope_utilisation"] = used / cap_usd if cap_usd else None
            raw = m.dual.get(m.f1_concessional_envelope_constraint, None)
            if raw is not None:
                val = -float(raw)
                if val < -ENVELOPE_BINDING_TOL:
                    out["envelope_dual_sign_anomaly"] = True
                val = max(0.0, val)
                out["f1_envelope_shadow_pv_per_undiscounted_usd"] = val
                out["f1_envelope_n_evaluable"] = 1
                out["f1_envelope_n_binding"] = int(val > ENVELOPE_BINDING_TOL)
    except Exception:
        pass
    return out


def parse_args():
    # docs/task_runner_repair.md, Step Three: this script had NO argparse
    # at all before this edit -- its load_scenario() call below either sat
    # on a hardcoded literal (gas_deliverability_case="baseline",
    # om_case="om_central") or omitted a sentinel-required key outright,
    # which would have raised a TypeError before ever reaching
    # build_model(). Built fresh, add_standard_arms(parser) for the full
    # set; capital_case stays sourced from FINANCING_ARMS (the runner's
    # own subject), not a flag, matching 01b_'s own convention.
    parser = argparse.ArgumentParser(
        description="18_run_pareto_frontier: cost-emissions Pareto frontier (MIX-2)"
    )
    # task_epsilon_constraint, Part 6B, Part 3: required, no default, no
    # glob, no fallback to a bare path -- see the module docstring's
    # "BASELINE DENOMINATOR" section for why. type=str (not Path) so the
    # value round-trips into output rows as a plain string, matching how
    # every other CLI-sourced value in this project's output is recorded.
    parser.add_argument(
        "--baseline-diagnostics",
        required=True,
        type=str,
        help="Path to a sliced, conversion-active baseline diagnostics "
             "JSON (01b_run_baseline_sliced.py's own hash8-tagged output "
             "-- e.g. results/baseline_sliced/diagnostics__<hash8>.json). "
             "No default, no glob: a runner silently selecting its own "
             "input file by pattern is the same silent-default class "
             "this project has already closed four times (the "
             "65,000,000 USD/TWh tariff default, storage_duration_hours, "
             "om_case hardcoded across six library functions, "
             "reserve_margin). Validated on load -- see "
             "_validate_baseline_diagnostics().",
    )
    add_standard_arms(parser)
    return parser.parse_args()


def _validate_baseline_diagnostics(baseline_diag, path):
    """
    task_epsilon_constraint, Part 6B, Part 3: confirm the file given to
    --baseline-diagnostics is a sliced, conversion-active baseline, not
    (a) an annual-builder diagnostics file, or (b) a sliced-but-
    conv_none one that would defeat this port's own purpose. Raises
    ValueError with a clear message if not.

    Field used: arm_tag, parsed. TASK.md offered three candidate
    signals (conversion_case, arm_tag, presence of a conversion
    diagnostic) -- only arm_tag is actually usable here. Read directly
    (scripts/01b_run_baseline_sliced.py): its DIAGNOSTICS file (the one
    --baseline-diagnostics must point at, since it alone carries the
    per-year emissions_tco2_by_year series this runner needs) carries
    "arm_tag" but NOT "conversion_case" or a "conversion" diagnostic --
    those two live only in 01b_'s separate SUMMARY file, which has no
    per-year series at all. 01b_'s tag format (its own source, read
    directly): "{peak_case}__{storage_cc_case}__{gas_build_case}__
    {new_gas_deliverability}__{gas_availability_case}__
    {gas_retirement_case}__{gas_capex_case}__{ccgt_efficiency_case}__
    {capital_recovery_case}__{dscr_case}__{capital_treatment_case}__
    {concessional_envelope_case}__{gas_deliverability_case}__
    {conversion_case}__{hydro_case}" -- conversion_case is "__"-joined
    segment index 13 (0-indexed) of 15.

    ONE check suffices for BOTH sliced-ness and conversion-activeness,
    not two: the ANNUAL builder raises ValueError on any conversion_case
    != "conv_none" (this port's own module docstring), so no annual-
    origin diagnostics file can EVER carry a non-"conv_none" conversion
    segment -- a file that passes "conversion segment != conv_none"
    could only have come from build_model_sliced(). Checking
    conversion-activeness therefore also proves sliced origin; a
    separate "is this sliced" test would be redundant, not additional
    safety.

    Fragility, stated rather than hidden: this parses a "__"-joined
    string by fixed position. It breaks if 01b_'s own tag format ever
    changes (e.g. a new arm inserted before conversion_case) --
    01b_run_baseline_sliced.py is outside this task's whitelist, so it
    cannot be given a cleaner top-level conversion_case field here. A
    follow-on adding that field to 01b_'s DIAGNOSTICS output (not just
    its summary) would let this function stop parsing a string.
    """
    arm_tag = baseline_diag.get("arm_tag")
    if not arm_tag:
        raise ValueError(
            f"--baseline-diagnostics {path!r} has no 'arm_tag' field. "
            "This does not look like 01b_run_baseline_sliced.py's own "
            "diagnostics output (which always carries arm_tag) -- most "
            "likely this is 01_run_baseline.py's (annual) output "
            "instead, which this runner can no longer use (the annual "
            "builder cannot represent conversion; see this module's own "
            "docstring)."
        )
    parts = arm_tag.split("__")
    if len(parts) < 15:
        raise ValueError(
            f"--baseline-diagnostics {path!r} has an arm_tag "
            f"({arm_tag!r}) with {len(parts)} '__'-joined segments, "
            "expected 15 (01b_run_baseline_sliced.py's own tag format). "
            "Cannot locate the conversion_case segment; refusing to "
            "guess."
        )
    conversion_segment = parts[13]
    if conversion_segment == "conv_none":
        raise ValueError(
            f"--baseline-diagnostics {path!r} is a sliced baseline but "
            f"its conversion_case is 'conv_none' (arm_tag={arm_tag!r}). "
            "This port exists so this runner can be compared against a "
            "CONVERSION-ACTIVE baseline (conv_central/conv_low/"
            "conv_high) -- a conv_none denominator defeats that "
            "purpose. Re-run 01b_run_baseline_sliced.py with an active "
            "--conversion-case and point --baseline-diagnostics at its "
            "output."
        )
    return arm_tag


def compute_endpoint_summary(df):
    """
    Per-arm endpoint characterisation from the raw point table. Pure
    (no solve, no file IO) so it can be run against a stored
    pareto_raw.csv. `df` may lack the TASK 26 columns (`point_type`);
    every row is then treated as a grid point and no anchor exists --
    this is the regression path against the pre-TASK-26 output.

    The first five keys per arm keep their pre-TASK-26 meaning and
    derivation exactly. TASK 26, Part B adds the rest; see
    docs/task_log_26b_pareto_anchors.md.
    """
    if "point_type" not in df.columns:
        df = df.assign(point_type="grid")
    summary = {}
    for label in list(dict.fromkeys(df["arm"])):
        arm_df = df[df["arm"] == label]
        pts = arm_df[arm_df["point_type"] != "unconstrained_anchor"]
        anchor = arm_df[arm_df["point_type"] == "unconstrained_anchor"]
        sub = pts[pts["status"] == "optimal"].sort_values("cap_fraction")
        binding_sub = sub[sub["cap_binding"] == True]
        non_binding_sub = sub[sub["cap_binding"] == False]

        # --- legacy keys, unchanged -----------------------------------
        frontier_begins_at = (
            float(binding_sub["cap_fraction"].max()) if len(binding_sub) else None
        )
        legacy_unconstrained = (
            float(non_binding_sub.sort_values("cap_fraction", ascending=False)
                  ["cum_emissions"].iloc[0])
            if len(non_binding_sub) else None
        )
        non_monotone = False
        if frontier_begins_at is not None:
            looser_but_binding = binding_sub[binding_sub["cap_fraction"] > frontier_begins_at]
            tighter_but_slack = non_binding_sub[non_binding_sub["cap_fraction"] < frontier_begins_at]
            non_monotone = len(looser_but_binding) > 0 or len(tighter_but_slack) > 0
        infeasible_sub = arm_df[arm_df["status"] != "optimal"]
        tightest_feasible_frac = float(sub["cap_fraction"].min()) if len(sub) else None
        loosest_infeasible_frac = (
            float(infeasible_sub["cap_fraction"].max()) if len(infeasible_sub) else None
        )

        # --- TASK 26, Part B ------------------------------------------
        anchor_ok = len(anchor) > 0 and anchor.iloc[0]["status"] == "optimal"
        if anchor_ok:
            a = anchor.iloc[0]
            unconstrained = float(a["cum_emissions"])
            unconstrained_source = "anchor_solve"
            peak = float(a["peak_capped_emissions_tco2"])
            base_annual = None
            g = pts[pts["cap_fraction"].notna() & pts["annual_cap_tco2"].notna()]
            if len(g):
                base_annual = float(g.iloc[0]["annual_cap_tco2"] / g.iloc[0]["cap_fraction"])
            loose_frac = (peak / base_annual) if base_annual else None
        else:
            unconstrained = legacy_unconstrained
            unconstrained_source = (
                "loosest_nonbinding_grid_point" if legacy_unconstrained is not None else None
            )
            peak, loose_frac = None, None
        grid_pts = pts[pts["point_type"] == "grid"]
        ext_pts = pts[pts["point_type"] == "extension"]
        loose_boundary = float(grid_pts["cap_fraction"].max()) if len(grid_pts) else None
        tight_boundary = float(pts["cap_fraction"].min()) if len(pts) else None
        tight_end_located = bool((pts["status"] == "infeasible").any())
        loose_end_located = bool(anchor_ok)
        reasons = []
        if not loose_end_located:
            reasons.append("loose end unlocated: no unconstrained anchor solve")
        if not tight_end_located:
            reasons.append(
                "tight end unlocated: every attempted point down to "
                f"cap_fraction={tight_boundary} was feasible"
            )
        within_grid = (
            bool(loose_frac <= loose_boundary + 1e-9)
            if (loose_frac is not None and loose_boundary is not None) else None
        )
        notes = []
        if loose_end_located and within_grid is False:
            # Located (the anchor solve measured it) but not gridded: not
            # a degeneracy, a statement about coverage.
            notes.append(
                f"loose end (cap_fraction={loose_frac:.4f}) lies ABOVE the grid's "
                f"loosest point ({loose_boundary}); the segment between them is "
                "not gridded"
            )
        n_off = (
            int((grid_pts["cap_fraction"] > loose_frac + 1e-9).sum())
            if loose_frac is not None else None
        )
        summary[label] = {
            "frontier_begins_at_cap_fraction":        frontier_begins_at,
            "unconstrained_optimum_emissions_tco2":   unconstrained,
            "tightest_feasible_cap_fraction":         tightest_feasible_frac,
            "loosest_infeasible_cap_fraction":        loosest_infeasible_frac,
            "binding_non_monotone":                   non_monotone,
            # --- TASK 26, Part B.1/B.3 ---
            "unconstrained_optimum_source":           unconstrained_source,
            "unconstrained_anchor_status":            (
                str(anchor.iloc[0]["status"]) if len(anchor) else None),
            "unconstrained_peak_capped_emissions_tco2": peak,
            "loose_end_cap_fraction":                 loose_frac,
            "loose_end_located":                      loose_end_located,
            "loose_end_within_grid":                  within_grid,
            "loose_boundary_cap_fraction":            loose_boundary,
            "n_points_at_loose_boundary":             int((grid_pts["cap_fraction"] == loose_boundary).sum()) if len(grid_pts) else 0,
            "n_grid_points_off_frontier":             n_off,
            "n_grid_points_on_frontier":              (int(len(grid_pts) - n_off) if n_off is not None else None),
            "tight_end_located":                      tight_end_located,
            "tight_boundary_cap_fraction":            tight_boundary,
            "n_points_at_tight_boundary":             int((pts["cap_fraction"] == tight_boundary).sum()) if len(pts) else 0,
            "n_extension_points_attempted":           int(len(ext_pts)),
            "n_extension_points_planned":             len(TIGHT_END_EXTENSION),
            "tight_end_floor_reached":                bool(
                (not tight_end_located)
                and tight_boundary is not None
                and abs(tight_boundary - min(TIGHT_END_EXTENSION)) < 1e-12
            ),
            # True when EITHER endpoint is unlocated.
            "endpoints_degenerate":                   bool(reasons),
            "endpoint_degeneracy_reasons":            reasons,
            "endpoint_notes":                         notes,
        }
    degenerate = [k for k, v in summary.items() if v["endpoints_degenerate"]]
    summary["_endpoint_degeneracy"] = {
        "n_arms_total": len(summary),
        "n_arms_degenerate": len(degenerate),
        "arms_degenerate": degenerate,
        # True when EITHER endpoint is unlocated for ANY arm: a frontier
        # reported without its endpoints established must say so.
        "any_endpoint_unlocated": bool(degenerate),
    }
    return summary


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    standard_kwargs = standard_arms_kwargs(args)

    # ── Load baseline emissions ───────────────────────────────
    # task_epsilon_constraint, Part 6B, Part 3: required path, no
    # fallback -- see module docstring "BASELINE DENOMINATOR".
    baseline_diagnostics_path = args.baseline_diagnostics
    with open(baseline_diagnostics_path) as f:
        baseline_diag = json.load(f)

    baseline_arm_tag = _validate_baseline_diagnostics(
        baseline_diag, baseline_diagnostics_path
    )

    baseline_emissions_by_year = baseline_diag["emissions_tco2_by_year"]
    years_str = sorted(baseline_emissions_by_year.keys())
    baseline_total = sum(baseline_emissions_by_year[y] for y in years_str)
    n_years = len(years_str)

    # Annual baseline emissions (roughly constant)
    baseline_annual = baseline_total / n_years

    print(f"\nPareto Frontier Sweep")
    print(f"  Baseline diagnostics: {baseline_diagnostics_path}")
    print(f"  Baseline arm_tag:     {baseline_arm_tag}")
    print(f"  Baseline total emissions: {baseline_total/1e6:.1f} MtCO2")
    print(f"  Baseline annual: {baseline_annual/1e6:.3f} MtCO2/yr")
    print(f"  Points: {N_POINTS} grid + up to {len(TIGHT_END_EXTENSION)} tight-end "
          f"extension (stop at first infeasible) + 1 unconstrained anchor, per arm")
    print(f"  Arms: {len(FINANCING_ARMS)}")
    print(f"  Solves: {len(FINANCING_ARMS) * (1 + N_POINTS)} minimum-plus-extension "
          f"(>= {len(FINANCING_ARMS) * (1 + N_POINTS + 1)}, <= "
          f"{len(FINANCING_ARMS) * (1 + N_POINTS + len(TIGHT_END_EXTENSION))})")

    # ── Generate cap levels ───────────────────────────────────
    # Sweep from 100% of baseline (unconstrained) to 10% of baseline
    cap_fractions = np.linspace(1.0, 0.10, N_POINTS)
    # TASK 26, Part B: per arm, the unconstrained anchor first (no cap at
    # all -- build_model_sliced already treats emissions_cap_by_year=None
    # and emissions_cap=None as "no constraint", the same path
    # 19_run_carbon_price_sweep.py uses), then the original grid, then
    # the tight-end extension.
    points = (
        [("unconstrained_anchor", None)]
        + [("grid", float(f)) for f in cap_fractions]
        + [("extension", float(f)) for f in TIGHT_END_EXTENSION]
    )

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    all_rows = []

    for arm in FINANCING_ARMS:
        print(f"\n  == {arm['label']} ==")

        stop_arm = False
        for i, (point_type, frac) in enumerate(points):
            if stop_arm:
                break
            is_anchor = point_type == "unconstrained_anchor"
            # Build uniform annual cap at this fraction of baseline
            # (anchor: no cap -- annual_cap/cap_by_year are None).
            annual_cap = None if is_anchor else baseline_annual * frac
            cap_by_year = None if is_anchor else [annual_cap] * n_years

            scenario = load_scenario(
                demand_level_case="served",
                demand_case="organic_central",
                capital_case=arm["capital_case"],
                solar_build_case="deployment_unconstrained",
                land_case="loose",
                carbon_case="no_policy",
                start_year=2025,
                end_year=MODEL_END_YEAR,
                **standard_kwargs,
            )
            scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
            scenario["financing_regime"] = arm["financing_regime"]
            # TASK 29: no required_margin override (dscr_case-derived; L34).

            years = scenario["years"]
            years_int = [int(y) for y in years]
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            try:
                # task_epsilon_constraint, Part 6B, Part 2: build_model ->
                # build_model_sliced -- keyword arguments unchanged, per
                # the port inventory (Part 6). This is the ONE line that
                # changes what this runner solves WITH; nothing about
                # what it measures or reports (Parts 2-4's own additions)
                # is touched.
                m = build_model_sliced(
                    scenario=scenario,
                    econ=econ,
                    emissions_cap_by_year=cap_by_year,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m, scenario=scenario)

                if not status["optimal"]:
                    if point_type == "extension":
                        stop_arm = True   # first infeasibility located the tight end
                    all_rows.append({
                        "arm": arm["label"], "cap_fraction": frac,
                        "point_type": point_type,
                        "annual_cap_tco2": annual_cap,
                        "status": "infeasible",
                        "leakage_share": None, "leakage_flagged": None,
                        # column-consistency with the "optimal" row shape
                        # below -- Part 2.2/2.1 fields are meaningless at
                        # an infeasible point, not silently omitted.
                        "cumulative_cap_tco2": (None if is_anchor else annual_cap * len(years_int)),
                        "cap_binding": None, "cap_slack_tco2": None,
                        "cap_binding_tolerance_tco2": None,
                        "n_binding_years": None, "n_years": len(years_int),
                        "carbon_shadow_pv_usd_per_tco2_binding_mean": None,
                        "carbon_shadow_yr_usd_per_tco2_binding_mean": None,
                        "carbon_shadow_pv_usd_per_tco2_simple_mean": None,
                        "carbon_shadow_yr_usd_per_tco2_simple_mean": None,
                        "carbon_shadow_yr_usd_per_tco2_discounted_mean": None,
                        "n_valid_dual_years": None,
                        "carbon_shadow_pv_by_year_json": None,
                        "carbon_shadow_yr_by_year_json": None,
                        # task_epsilon_constraint, Part 6B, Part 3: record
                        # the baseline dependency in every row, not only
                        # optimal ones.
                        "baseline_diagnostics_path": str(baseline_diagnostics_path),
                        "baseline_arm_tag": baseline_arm_tag,
                    })
                    print(f"    {'unconstrained' if is_anchor else format(frac, '.0%')} "
                          f"{'' if is_anchor else 'cap '}-> INFEASIBLE")
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                decomp = diag.get("cost_decomposition", {})

                total_cost = float(pyo.value(m.system_cost_npv))
                cum_emissions = sum(diag["emissions_tco2_by_year"].values())
                cum_unserved = sum(diag["unserved_twh_by_year"].values())
                cum_demand = sum(diag["demand_twh_by_year"].values())
                cum_generation = cum_demand - cum_unserved

                # ── task_epsilon_constraint, Part 2.1: read the dual that
                # is ALREADY there -- extract_planning_diagnostics() (just
                # called above) already computes the emissions-constraint
                # dual; no new extractor is built here. ──
                carbon_shadow_by_year = diag["carbon_shadow_pv_usd_per_tco2_by_year"]
                carbon_shadow_by_year_yr = diag["carbon_shadow_yr_usd_per_tco2_by_year"]

                # ── Part 2.2: binding check. Tolerance RELATIVE to the
                # cap, not absolute -- the same reasoning D1's fix applied
                # to the interaction classifier (task_defect_closure): an
                # absolute tCO2 tolerance would fail across this sweep's
                # own range (10% to 100% of baseline annual emissions).
                # Per-year slack, MINIMUM across years -- not a
                # cumulative/average test. This project has already found
                # (extract_reliability_dual's own comment, for the grid
                # service obligation) that an aggregate test can read
                # "not binding" while individual years sit pinned exactly
                # on the bound; the cap here is FLAT (annual_cap repeated
                # every year), so "the tightest year" is well-defined and
                # cheap to compute directly, without needing an aggregate
                # proxy at all.
                CAP_BINDING_REL_TOL = 0.01  # 1% of the annual cap
                cap_tolerance_tco2 = (
                    None if is_anchor else CAP_BINDING_REL_TOL * annual_cap
                )
                per_year_slack_tco2 = [] if is_anchor else [
                    annual_cap - diag["emissions_tco2_by_year"][y]
                    for y in years_int
                ]
                cap_slack_tco2 = min(per_year_slack_tco2) if per_year_slack_tco2 else None
                cap_binding = (
                    None if is_anchor else (cap_slack_tco2 <= cap_tolerance_tco2)
                )
                # TASK 26, Part B: the cap constrains m.emissions_capped_by_year
                # (GRID-ONLY by default), but cap_slack_tco2/cap_binding above
                # read diag["emissions_tco2_by_year"], which is TOTAL (grid +
                # self-generation). Where self-generation is large (the tight
                # public arm; leakage_share up to 0.95) the slack above can
                # read negative with the cap not actually binding. Kept
                # unchanged (meaning not altered); the quantity the cap
                # really constrains is recorded separately.
                capped_by_year = [
                    float(pyo.value(m.emissions_capped_by_year[t]))
                    for t in range(len(years_int))
                ]
                peak_capped_emissions_tco2 = max(capped_by_year)
                cap_slack_capped_tco2 = (
                    None if is_anchor else min(annual_cap - e for e in capped_by_year)
                )
                cap_binding_capped = (
                    None if is_anchor
                    else bool(cap_slack_capped_tco2 <= cap_tolerance_tco2)
                )

                # ── Part 2.1 continued: scalar reduction of the year-by-
                # year dual. THREE reductions computed (all cheap); the
                # one used for the frontier table is the BINDING-YEAR
                # MEAN (Zeus's stated preference, TASK.md 2.1): a
                # non-binding year's dual is 0 or None and would drag a
                # simple mean toward a number with no marginal-abatement-
                # cost meaning at that cap level. "Binding" for this
                # reduction reuses the IDENTICAL per-year tolerance as
                # cap_binding above, for consistency -- a year counts as
                # binding for the dual reduction iff it counts as binding
                # for the point's own cap_binding flag. None entries
                # (the extractor's own "no dual available" signal) are
                # excluded from every reduction, not treated as zero;
                # if EVERY year is None, the reduction is None, not a
                # fabricated number.
                binding_years = [
                    y for y, slack in zip(years_int, per_year_slack_tco2)
                    if slack <= cap_tolerance_tco2
                ]

                def _mean(vals):
                    vals = [v for v in vals if v is not None]
                    return float(np.mean(vals)) if vals else None

                def _discounted_mean(vals_by_year):
                    num, den = 0.0, 0.0
                    for t, y in enumerate(years_int):
                        v = vals_by_year.get(y)
                        if v is None:
                            continue
                        df_t = float(pyo.value(m.DF[t]))
                        num += df_t * v
                        den += df_t
                    return (num / den) if den > 0 else None

                # TASK 53 2.3 (register L106): every reduction on BOTH bases. The former
                # `discounted_mean` DF-weighted the PRESENT-VALUE series, which discounts
                # a number that is already discounted (an effective DF^2 weighting). The
                # discounted mean is now taken of the YEAR-t series only, where DF-weighting
                # is meaningful (it equals sum(pv) / sum(DF)).
                carbon_shadow_binding_mean = _mean(
                    [carbon_shadow_by_year.get(y) for y in binding_years]
                )
                carbon_shadow_binding_mean_yr = _mean(
                    [carbon_shadow_by_year_yr.get(y) for y in binding_years]
                )
                carbon_shadow_simple_mean = _mean(list(carbon_shadow_by_year.values()))
                carbon_shadow_simple_mean_yr = _mean(list(carbon_shadow_by_year_yr.values()))
                carbon_shadow_discounted_mean_yr = _discounted_mean(carbon_shadow_by_year_yr)
                n_binding_years = len(binding_years)
                n_valid_dual_years = sum(
                    1 for v in carbon_shadow_by_year.values() if v is not None
                )

                solar_total = sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                )
                storage_final = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))

                # task_epsilon_constraint, Part 6B, Part 2: m.solar_
                # generation[t] / m.gas_generation[t] / m.storage_
                # discharge[t] are Var(T) in the annual builder but
                # Var(T, S) (slice-indexed) in build_model_sliced() -- a
                # bare [t] index on the sliced Vars raises. Read the
                # already-slice-aggregated per-year series diag already
                # carries (extract_planning_diagnostics's own
                # _slice_total aggregation, confirmed builder-agnostic
                # in the Part 6 inventory) instead of indexing m
                # directly. On the annual builder this returns the
                # identical number pyo.value(m.x[t]) did (_slice_total's
                # own annual branch is exactly pyo.value(comp[t])), so
                # no existing annual result changes; it only stops
                # crashing on a sliced one -- this is what "confirm
                # nothing else blocks conv_central" (TASK.md Part 2.5)
                # found and required fixing.
                solar_gen_by_year = diag["solar_generation_twh_e_by_year"]
                # RENAMED from gas_generation_twh_e_by_year (task_conversion_
                # diagnostics, Part 2C) -- FLEET-ONLY, excludes m.convert_
                # generation. task_conversion_diagnostics Part 2C verdict for
                # THIS call site: "arithmetic now visibly wrong" -- under an
                # active conversion arm converted plant supplies most of
                # gas-fired electricity (88% at central), so disc_generation
                # below understates total generation and overstates
                # system_lcoe on any conversion-active run. NOT fixed here
                # (a rename must not also change a computation); reported to
                # the register instead.
                gas_gen_by_year = diag["fleet_gas_generation_twh_e_by_year"]
                storage_disch_by_year = diag["storage_discharge_twh_e_by_year"]

                # System LCOE: total discounted cost / total discounted generation
                disc_generation = sum(
                    float(pyo.value(m.DF[t])) * (
                        solar_gen_by_year[int(years[t])]
                        + gas_gen_by_year[int(years[t])]
                        + storage_disch_by_year[int(years[t])]
                    )
                    for t in range(len(years))
                )
                system_lcoe = total_cost / (disc_generation * 1e6) if disc_generation > 0 else None

                # Leakage (2.6, D15). On a deliberate cap sweep a flagged point
                # is DATA, not a warning: it marks where the frontier is met by
                # dumping load to diesel rather than decarbonising. Recorded as
                # a column, no print; mark or exclude flagged points when the
                # frontier is drawn.
                leak = leakage_diagnostic(m)

                all_rows.append({
                    **annual_emissions_flat(m, years),
                    "arm":              arm["label"],
                    "cap_fraction":     frac,
                    "point_type":       point_type,
                    "annual_cap_tco2":  annual_cap,
                    "status":           "optimal",
                    "npv_cost":         total_cost,
                    "real_expenditure": decomp.get("real_expenditure_npv", None),
                    "voll_penalty":     decomp.get("voll_penalty_npv", None),
                    "cum_emissions":    cum_emissions,
                    "cum_unserved":     cum_unserved,
                    "solar_total_mw":   solar_total,
                    "storage_mwh":      storage_final,
                    "system_lcoe":      system_lcoe,
                    # task_epsilon_constraint, Part 6B, Part 2: read from
                    # the slice-aggregated dicts above, not m directly --
                    # see the comment at their computation.
                    "gas_share_2030":   (
                        gas_gen_by_year[int(years[5])]
                        / max(solar_gen_by_year[int(years[5])]
                              + gas_gen_by_year[int(years[5])]
                              + storage_disch_by_year[int(years[5])], 1e-9)
                    ),
                    "solar_share_2045": (
                        solar_gen_by_year[int(years[len(years)-1])]
                        / max(solar_gen_by_year[int(years[len(years)-1])]
                              + gas_gen_by_year[int(years[len(years)-1])]
                              + storage_disch_by_year[int(years[len(years)-1])], 1e-9)
                    ),
                    "leakage_share":    leak["leakage_share"],
                    "leakage_flagged":  leak["flagged"],
                    # ── Part 2.2: binding check ──
                    "cumulative_cap_tco2":        (None if is_anchor else annual_cap * len(years_int)),
                    "cap_binding":                cap_binding,
                    "cap_slack_tco2":             cap_slack_tco2,
                    # TASK 26, Part B: additive -- the capped (grid-only) quantity.
                    "peak_capped_emissions_tco2":     peak_capped_emissions_tco2,
                    "cap_slack_capped_quantity_tco2": cap_slack_capped_tco2,
                    "cap_binding_capped_quantity":    cap_binding_capped,
                    "cap_binding_tolerance_tco2": cap_tolerance_tco2,
                    "n_binding_years":            n_binding_years,
                    "n_years":                    len(years_int),
                    # ── Part 2.1: the dual, reduced three ways, plus the
                    # full year-by-year series (CSV runner -> flat field,
                    # JSON-encoded, matching this project's existing
                    # convention for a per-year series inside a CSV row
                    # -- e.g. 15_run_pol1_ndc_comparison.py's
                    # "solar_add_by_year_json").
                    "carbon_shadow_pv_usd_per_tco2_binding_mean":     carbon_shadow_binding_mean,
                    "carbon_shadow_yr_usd_per_tco2_binding_mean":     carbon_shadow_binding_mean_yr,
                    "carbon_shadow_pv_usd_per_tco2_simple_mean":      carbon_shadow_simple_mean,
                    "carbon_shadow_yr_usd_per_tco2_simple_mean":      carbon_shadow_simple_mean_yr,
                    "carbon_shadow_yr_usd_per_tco2_discounted_mean":  carbon_shadow_discounted_mean_yr,
                    "n_valid_dual_years":           n_valid_dual_years,
                    "carbon_shadow_pv_by_year_json":  json.dumps(carbon_shadow_by_year),
                    "carbon_shadow_yr_by_year_json":  json.dumps(carbon_shadow_by_year_yr),
                    # task_epsilon_constraint, Part 6B, Part 3.
                    "baseline_diagnostics_path": str(baseline_diagnostics_path),
                    "baseline_arm_tag": baseline_arm_tag,
                    # TASK 30, Part A: additive; nothing above is altered.
                    **_envelope_fields(m, diag, scenario, range(len(years_int))),
                })

                print(f"    {'uncon' if is_anchor else format(frac, '>4.0%')} cap -> cost=${total_cost/1e9:.1f}B  "
                      f"emit={cum_emissions/1e6:.1f}Mt  "
                      f"unserved={cum_unserved:.1f}TWh  "
                      f"solar={solar_total/1e3:.1f}GW")

            except Exception as e:
                all_rows.append({
                    "arm": arm["label"], "cap_fraction": frac,
                    "point_type": point_type,
                    "annual_cap_tco2": annual_cap, "status": f"error: {e}",
                    "leakage_share": None, "leakage_flagged": None,
                    "cumulative_cap_tco2": (None if is_anchor else annual_cap * len(years_int)),
                    "cap_binding": None, "cap_slack_tco2": None,
                    "cap_binding_tolerance_tco2": None,
                    "n_binding_years": None, "n_years": len(years_int),
                    "carbon_shadow_pv_usd_per_tco2_binding_mean": None,
                    "carbon_shadow_yr_usd_per_tco2_binding_mean": None,
                    "carbon_shadow_pv_usd_per_tco2_simple_mean": None,
                    "carbon_shadow_yr_usd_per_tco2_simple_mean": None,
                    "carbon_shadow_yr_usd_per_tco2_discounted_mean": None,
                    "n_valid_dual_years": None,
                    "carbon_shadow_pv_by_year_json": None,
                    "carbon_shadow_yr_by_year_json": None,
                    "baseline_diagnostics_path": str(baseline_diagnostics_path),
                    "baseline_arm_tag": baseline_arm_tag,
                })
                print(f"    {'uncon' if is_anchor else format(frac, '.0%')} cap -> ERROR: {e}")

    # ── Save ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # TASK 26, Part B: endpoint characterisation, computed by the pure
    # compute_endpoint_summary() (runnable against a stored pareto_raw.csv
    # with no solve). `on_frontier` is the row-level answer to "is this
    # point on the frontier or does it merely sit on the grid": a grid or
    # extension point is on the frontier iff its cap is TIGHTER than the
    # unconstrained optimum's own peak capped emissions (the cap fraction
    # at which a flat cap stops binding). None where no anchor exists.
    endpoint_summary = compute_endpoint_summary(df)
    df["on_frontier"] = None
    for label in [a_["label"] for a_ in FINANCING_ARMS]:
        loose = endpoint_summary[label]["loose_end_cap_fraction"]
        if loose is None:
            continue
        m_ = (df["arm"] == label) & (df["point_type"] != "unconstrained_anchor")
        df.loc[m_, "on_frontier"] = df.loc[m_, "cap_fraction"] <= loose + 1e-9
    df.to_csv(RESULTS_DIR / "pareto_raw.csv", index=False)

    optimal_all = df[df["status"] == "optimal"].copy()
    anchors_df = optimal_all[optimal_all["point_type"] == "unconstrained_anchor"]
    # Grid + extension points only: the anchor is not a cap point and
    # must not enter the cap-indexed logic below.
    optimal = optimal_all[optimal_all["point_type"] != "unconstrained_anchor"].copy()

    print(f"\n{'='*90}")
    print(f"  FRONTIER A ENDPOINTS (per financing arm)")
    print(f"{'='*90}")
    for arm in FINANCING_ARMS:
        label = arm["label"]
        e = endpoint_summary[label]
        uc = e["unconstrained_optimum_emissions_tco2"]
        lf = e["loose_end_cap_fraction"]
        uc_str = (f"{uc/1e6:.1f}Mt ({e['unconstrained_optimum_source']})"
                  if uc is not None else "n/a")
        lf_str = f"{lf:.4f}" if lf is not None else "UNLOCATED"
        above = "" if e["loose_end_within_grid"] in (True, None) else " (ABOVE the grid)"
        if e["tight_end_located"]:
            tight_str = (f"located (first infeasible at "
                         f"{e['loosest_infeasible_cap_fraction']:g})")
        else:
            tight_str = (f"NOT LOCATED (feasible down to "
                         f"{e['tight_boundary_cap_fraction']:g})")
        print(f"  {label:<25} unconstrained emissions={uc_str}  "
              f"frontier begins at cap_fraction<={lf_str}{above}  "
              f"tightest feasible={e['tightest_feasible_cap_fraction']}  "
              f"tight end: {tight_str}")
        print(f"    grid points on/off frontier: "
              f"{e['n_grid_points_on_frontier']}/{e['n_grid_points_off_frontier']}  "
              f"points at loose boundary={e['n_points_at_loose_boundary']}  "
              f"at tight boundary={e['n_points_at_tight_boundary']}  "
              f"extension attempted {e['n_extension_points_attempted']}/{e['n_extension_points_planned']}")
        for note in e["endpoint_notes"]:
            print(f"    NOTE: {note}")
        if e["endpoints_degenerate"]:
            print(f"    ENDPOINTS DEGENERATE: " + "; ".join(e["endpoint_degeneracy_reasons"]))
        if e["binding_non_monotone"]:
            print(f"    WARNING: binding/non-binding split is NOT monotone "
                  f"in cap_fraction for {label} -- see pareto_raw.csv, "
                  f"cap_binding column.")
    deg = endpoint_summary["_endpoint_degeneracy"]
    print(f"\n  ENDPOINT DEGENERACY: {deg['n_arms_degenerate']} of {deg['n_arms_total']} arms "
          f"have an unlocated endpoint"
          f"{' -- ' + ', '.join(deg['arms_degenerate']) if deg['arms_degenerate'] else ''}. "
          f"{'This frontier is reported WITHOUT its endpoints established.' if deg['any_endpoint_unlocated'] else 'Both endpoints are located for every arm.'}")

    # ── Non-dominated check (TASK 30, Part B). Standard Pareto dominance
    # (j dominates i if cost and emissions are both <=, one strictly <),
    # checked per arm among ALL optimal points INCLUDING the uncapped anchor
    # as a candidate dominator, on BOTH emissions axes (total and grid-only);
    # src/pareto_plot.py holds the definition so the figure and this check
    # cannot disagree. ──────────────────────────────────────────────────
    print(f"\n{'='*90}\n  NON-DOMINATED CHECK\n{'='*90}")
    flagged = flag_dominance(pd.DataFrame(all_rows))
    for arm in FINANCING_ARMS:
        g = flagged[flagged["arm"] == arm["label"]]
        print(f"  {arm['label']:<22} non-dominated on total emissions: "
              f"{int((~g['dominated_total']).sum())} of {len(g)}; on grid-only emissions: "
              f"{int((~g['dominated_grid_only']).sum())} of {len(g)}")
    dom_df = dominated_points_table(flagged)
    if len(dom_df):
        print(f"  {len(dom_df)} point(s) dominated on TOTAL emissions. {DOMINATED_NOTE}")
        dom_df.to_csv(RESULTS_DIR / "pareto_dominated_points.csv", index=False)
    else:
        print("  None found -- every reported optimal point is non-dominated.")

    with open(RESULTS_DIR / "pareto_endpoints.json", "w") as f:
        json.dump(endpoint_summary, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'pareto_endpoints.json'}")

    # ── Pareto frontier figure (TASK 30, Part B): two panels, grid-only
    # and total emissions; solid = non-dominated, hollow = dominated. ──
    draw_two_panel(flagged, RESULTS_DIR / "pareto_frontier.png",
                   RESULTS_DIR / "pareto_frontier.pdf")

    # ── Optimal mix table at key cap levels ───────────────────
    # task_epsilon_constraint, Part 2.5: flag, don't drop -- a row here
    # whose cap is not binding is annotated "(non-binding)" rather than
    # silently presented as a point on the frontier. At cap_fraction=1.0
    # in particular, the cap is essentially the unconstrained baseline
    # and is very likely non-binding for every arm.
    key_fracs = [1.0, 0.80, 0.60, 0.40, 0.20]
    print(f"\n{'='*90}")
    print(f"  GENERATION MIX AT KEY CAP LEVELS (* = cap NOT binding, not a frontier point)")
    print(f"{'='*90}")
    print(f"  {'Arm':<25} {'Cap':>5} {'Cost($B)':>10} {'Emit(Mt)':>10} "
          f"{'Solar(GW)':>10} {'Stor(GWh)':>10} {'Unserved':>10} {'LCOE':>8}")
    print(f"  {'-'*88}")

    for arm_label in arm_styles:
        for frac in key_fracs:
            row = optimal[
                (optimal["arm"] == arm_label)
                & (np.abs(optimal["cap_fraction"] - frac) < 0.02)
            ]
            if len(row):
                r = row.iloc[0]
                lcoe_str = f"${r['system_lcoe']:.1f}" if r['system_lcoe'] else "n/a"
                _on = r["on_frontier"] if pd.notna(r["on_frontier"]) else r["cap_binding"]
                flag = "" if _on else "*"
                print(f"  {arm_label:<25} {frac:>4.0%}{flag:<1} "
                      f"${r['npv_cost']/1e9:>8.1f}B "
                      f"{r['cum_emissions']/1e6:>9.1f} "
                      f"{r['solar_total_mw']/1e3:>9.1f} "
                      f"{r['storage_mwh']/1e3:>9.1f} "
                      f"{r['cum_unserved']:>9.1f} "
                      f"{lcoe_str:>8}")

    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
