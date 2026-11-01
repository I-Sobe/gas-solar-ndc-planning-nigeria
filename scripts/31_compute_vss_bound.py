"""
31_compute_vss_bound.py  —  the VSS-partial bound, ECP_free vs WS
==========================================================================

TASK 06, Part 3 (original four guards); TASK 08, Part D (fifth guard,
see Section 3 below). Reads 30_run_stochastic_redispatch.py's output
(the ECP_free leg, "expected cost of the central plan") and
17_run_monte_carlo.py's output (the WS leg, "wait-and-see": full
re-optimisation per draw) and, ONLY if five named guards all pass,
emits:

    vss_partial  in  [0, bound_upper],   bound_upper = ECP_free - WS

1. WHY "ECP", NEVER "EEV", AND WHY THE FIRST STAGE IS SOLAR+STORAGE ONLY
-------------------------------------------------------------------------
`firm_capacity_constraint` has no priced slack (TASK 01, item 19).
Under `--fix-gas fixed`, one infeasible draw makes the Stage-2
expectation infinite and the bound degenerates. Under `--fix-gas free`,
`gas_add` adapts per draw -- which makes that leg MORE adaptive than a
textbook recourse problem whose first stage includes gas, not less.
The only internally consistent reading is: the first stage is solar and
storage ONLY; gas build and dispatch are second-stage recourse in both
legs. Under that reading:

    WS  <=  RP_partial  <=  ECP_free

RP_partial (solar+storage fixed, gas optimised jointly with the OTHER
draws' gas decisions under full stochastic recourse) is never solved
here -- building it would require editing the builder to accept a
multi-scenario recourse formulation, explicitly out of scope. So:

    VSS_partial  in  [0, ECP_free - WS]

The upper end also contains EVPI (the value of knowing gas regime
before committing solar+storage), inseparable from VSS without
RP_partial. This script never computes, prints, or names anything
"EEV": 30_'s fixed plan comes from the central arm string (a discrete
CLI configuration), not a mean over a distribution, and the gas regime
is itself categorical -- "expected value of the expected value" has no
referent here. The quantity's name, everywhere in this script's output,
is `ecp` / `ecp_free_usd`.

2. THE GUARD THAT MATTERS MOST -- read, not inferred from "free"
-------------------------------------------------------------------------
`scripts/30_run_stochastic_redispatch.py`'s `--fix-gas` branching
(confirmed by reading the file, not the flag's name) has an `if
args.fix_gas == "fixed": ...` block that fixes or bands `m.gas_add[t]`
directly on the built model. There is NO corresponding `elif
args.fix_gas == "free":` branch -- under `free`, NOTHING executes on
`m.gas_add`. So `--fix-gas free` itself applies zero constraint of its
own. Whatever bound (if any) exists on `gas_add` comes entirely from
`gas_build_case`, a SEPARATE required standard arm, independent of
`--fix-gas`:

  - `gas_build_case="gas_unconstrained"` (`src/optimize_model_sliced.py`,
    "pass  # no cap; the LP builds whatever is cost-optimal"): `gas_add`
    is a plain `NonNegativeReals` Var with no finite upper bound at all.
    GENUINELY unbounded -- no band, nothing for RP_partial to replicate.
  - `gas_build_case="gas_ndc_17gw"`: a cap exists, but on CUMULATIVE
    `gas_capacity_mw[t]`, not a per-year band on `gas_add[t]` itself --
    structurally different from `--fix-gas fixed`'s per-year
    `[v*(1-tol), v*(1+tol)]` band. RP_partial, if built, would need the
    same cumulative cap, not a per-year band.
  - `gas_build_case="gas_none"`: `gas_add[t]` is fixed to 0 for all t --
    gas is not free at all under this arm, and the whole "gas as
    recourse" framing in Section 1 does not hold. This script treats
    this case as invalidating the construction, not merely banding it
    (see Guard 2b below).

This script reads the ACTUAL `gas_build_case` recorded in the Stage-1
summary being evaluated (not a hardcoded assumption) and reports which
of the three cases applies, every run.

3. FIVE GUARDS, EACH CHECKED AND REPORTED BY NAME -- refuses otherwise
-------------------------------------------------------------------------
  (1) same_draws_sha256   -- both legs' draws file match by SHA-256,
      not filename. 30_ does not store a draws-file reference in its
      own summary.json (the draws tag deliberately excludes
      --fix-gas/--fix-tolerance so one draws file serves both --fix-gas
      arms -- see 30_'s Part 1 commit); this script reconstructs 30_'s
      own draws_tag formula from the Stage-1 summary's recorded
      standard_arms/own_args, re-derives the hash8, and self-checks the
      reconstruction against the stored full tag in
      results/stochastic/draws__<hash8>_meta.json before trusting it.
      17_ did not record what it consumed either (fixed, unhashed
      output filenames, no arm tag) until TASK 06: see the
      "PROVENANCE ADDENDUM" note below.
  (2) fix_gas_free        -- Stage-1 summary's own_args.fix_gas == "free".
  (2b) gas_build_not_none -- Stage-1's gas_build_case != "gas_none"
      (Section 2's third bullet: "free" gas that is actually fixed to
      zero is not a recourse leg at all). Not one of TASK 06's four
      named guards by number, but folded into guard (2) since a
      "free" leg that cannot build any gas is not meaningfully
      different from "fixed" at v=0 -- reported as part of (2)'s detail,
      not a separate gate, so the original four-guard count stays
      traceable inside this one.
  (3) gas_deliverability_baseline -- Stage-1's standard_arms
      .gas_deliverability_case == "baseline" (TASK 01 item 18: nothing
      in the builder enforces that Stage 1's own deliverability regime
      is the MODE of the Stage-2 sampling distribution; enforced here).
  (4) ws_case_scenario_matches_stage1 -- TASK 08, Part D. The selected
      --ws-case must match 30_'s Stage-1 configuration on every field
      TASK 08 Part B's enumeration found differing: capital_case,
      financing_regime, required_margin, has_emissions_cap. TASK 06
      held this at "report the mismatch, do not gate on it"
      (`bound_is_sane`) because no 17_ case was scenario-comparable to
      30_'s central plan; TASK 08 Part B added a fifth, matched case
      (and fixed the required_margin/financing_regime literals the
      original four cases carried, which were themselves stale --
      dscr_scenarios()'s own docstring states the pre-F1 1.10/1.05
      required_margin split these cases hardcode was "removed" when
      dscr_case replaced it), making this a real refusal gate rather
      than a reported caveat. Reads both sides from provenance each
      script now records for exactly this purpose (30_'s
      stage1_scenario_fields, 17_'s case_fields) -- not re-derived or
      assumed here. Naming: TASK 08 calls this guard
      `ws_case_scenario_matches_stage1`; it is guard (4) in this
      script's own dict ordering (after the three from TASK 06),
      immediately before the feasibility guard below.
  (5) zero_infeasible_both_legs -- Stage-1 n_infeasible == 0 AND the
      selected WS case's n_infeasible == 0 (ws_valid == True).

PROVENANCE ADDENDA (not named A/B/C/D fixes in their own tasks, added
because the guard immediately above each would otherwise be
unverifiable):
  - Guard (1): 17_run_monte_carlo.py (TASK 06) writes
    results/monte_carlo/monte_carlo_meta.json (draws_file path + its own
    SHA-256, standard_arms, per-case ws_valid) immediately after
    eaas_necessity.json. Before this addition, 17_'s output had no arm
    tag and no record of which --draws-file produced it.
  - Guard (4): 30_ (TASK 08) adds summary.json's stage1_scenario_fields
    (capital_case, financing_regime, required_margin, has_emissions_cap
    -- none of these were CLI arms, so standard_arms/own_args never
    carried them); 17_ (TASK 08) adds monte_carlo_meta.json's
    case_fields (the same four fields, resolved per case, read directly
    off each case's own built scenario dict rather than re-deriving
    load_scenario()'s default logic a second time).

4. RESOLVED BY TASK 08 -- NO LONGER OPEN CAVEATS
-------------------------------------------------------------------------
TASK 06 reported three findings that conditioned whether ANY bound this
script computed was a valid WS/ECP_free comparison, without gating on
them (the task named exactly four guards at the time). TASK 08 resolved
all three in 17_run_monte_carlo.py directly -- `known_validity_caveats`
below is now empty, kept as a field (not removed) so a caller reading
this script's JSON output sees explicitly that the check still runs and
currently has nothing to report, rather than inferring that from the
field's absence:
  (A) peak_mw now recomputed per draw (TASK 08 Part C, copied verbatim
      from 30_'s own _recompute_peak_mw -- same source, not a second
      implementation).
  (B) 17_ now calls build_model_sliced, the same builder 30_ (ECP_free)
      uses (TASK 08 Part A).
  (C) a fifth, additive case in 17_'s CASES now matches 30_'s Stage-1
      configuration field-by-field (TASK 08 Part B) -- promoted from a
      reported caveat to guard (4) above, since it is now resolvable
      rather than merely diagnosable.

5. OUT OF SCOPE (unchanged by this script)
-------------------------------------------------------------------------
No change to either builder. No priced slack on firm_capacity_constraint.
No change to src/scenarios.py. No Monte Carlo run of any kind -- this
script only reads existing 30_/17_ output files.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output_naming import hash8_of, append_index

RESULTS_DIR = ROOT / "results" / "vss_bound"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# TASK 08: all three of TASK 06's reported caveats are now resolved in
# 17_run_monte_carlo.py (peak_mw recompute, sliced builder, a matched
# fifth case -- the third promoted to guard (4) above rather than left
# here). Kept as an explicit empty list, not removed, so this field's
# presence in the JSON output still says "checked, nothing outstanding"
# rather than its absence being read either way.
KNOWN_VALIDITY_CAVEATS = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="31_compute_vss_bound: ECP_free vs WS, VSS_partial upper bound"
    )
    parser.add_argument(
        "--stage1-summary", type=str, required=True,
        help="Path to 30_run_stochastic_redispatch.py's summary__<hash8>.json "
             "for the ECP_free leg (must have been run with --fix-gas free). "
             "Required, no default.",
    )
    parser.add_argument(
        "--mc-results-dir", type=str, required=True,
        help="Path to 17_run_monte_carlo.py's results directory (contains "
             "monte_carlo_summary.csv and monte_carlo_meta.json for the WS "
             "leg). Required, no default -- this script never assumes the "
             "live results/monte_carlo/ is the run intended.",
    )
    parser.add_argument(
        "--ws-case", type=str, required=True,
        help="Which row of 17_'s monte_carlo_summary.csv 'case' column to "
             "read as WS. Required, no default: Section 4(C) above means "
             "no case is a correct scenario match, so this script will not "
             "guess one on the caller's behalf.",
    )
    return parser.parse_args()


def reconstruct_draws_tag(stage1_summary: dict) -> str:
    """
    Re-derive 30_'s draws_tag formula (results/stochastic writer, Part 1)
    from the Stage-1 summary's own recorded standard_arms/own_args.
    30_'s summary.json carries no direct pointer to its draws file by
    design (the draws tag excludes --fix-gas/--fix-tolerance so one
    draws file serves both --fix-gas arms) -- this is the only way to
    recover it after the fact.
    """
    standard_arms = stage1_summary["standard_arms"]
    oa = stage1_summary["own_args"]
    standard_tag = "__".join(f"{k}={v}" for k, v in sorted(standard_arms.items()))
    return (
        f"{standard_tag}__n_draws={oa['n_draws']}__seed={oa['seed']}"
        f"__demand_sigma={oa['demand_sigma']}__carbon_mu={oa['carbon_mu']}"
        f"__carbon_sigma={oa['carbon_sigma']}__gas_weights={oa['gas_weights']}"
    )


def check_same_draws_sha256(stage1_summary: dict, mc_meta: dict) -> dict:
    expected_tag = reconstruct_draws_tag(stage1_summary)
    draws_hash8 = hash8_of(expected_tag)
    meta_path = ROOT / "results" / "stochastic" / f"draws__{draws_hash8}_meta.json"
    if not meta_path.exists():
        return {
            "passed": False,
            "reason": (f"Reconstructed draws_hash8={draws_hash8} from the Stage-1 "
                       f"summary's standard_arms/own_args, but "
                       f"{meta_path} does not exist. Either the tag-"
                       f"reconstruction formula in this script has drifted from "
                       f"30_'s own, or this Stage-1 summary predates Part 1's "
                       f"draws-sharing code."),
        }
    draws_meta = json.loads(meta_path.read_text())
    if draws_meta.get("draws_tag") != expected_tag:
        return {
            "passed": False,
            "reason": (f"Self-check failed: {meta_path} exists at the expected "
                       f"hash8 but its own recorded draws_tag does not match "
                       f"this script's reconstruction -- treat as a hash "
                       f"collision / formula drift, not a confirmed match. "
                       f"expected={expected_tag!r} found={draws_meta.get('draws_tag')!r}"),
        }
    stage1_sha = draws_meta["csv_sha256"]
    stage2_sha = mc_meta.get("draws_file_sha256")
    if stage2_sha is None:
        return {
            "passed": False,
            "reason": ("17_'s monte_carlo_meta.json has no draws_file_sha256 "
                       "-- re-run 17_run_monte_carlo.py with the current "
                       "version of the script (provenance addendum)."),
        }
    passed = stage1_sha == stage2_sha
    return {
        "passed": passed,
        "stage1_draws_sha256": stage1_sha,
        "stage2_draws_sha256": stage2_sha,
        "stage1_draws_csv": draws_meta["csv_path"],
        "stage2_draws_file": mc_meta.get("draws_file"),
        "reason": None if passed else "SHA-256 mismatch -- the two legs were not run on the same draws.",
    }


def check_fix_gas_free(stage1_summary: dict) -> dict:
    fix_gas = stage1_summary["own_args"]["fix_gas"]
    gas_build_case = stage1_summary["standard_arms"].get("gas_build_case")
    if fix_gas != "free":
        return {"passed": False, "fix_gas": fix_gas, "gas_build_case": gas_build_case,
                "reason": f"Stage-1 summary was run with --fix-gas {fix_gas!r}, not 'free'."}
    if gas_build_case == "gas_none":
        return {"passed": False, "fix_gas": fix_gas, "gas_build_case": gas_build_case,
                "reason": ("gas_build_case='gas_none' fixes gas_add to 0 for every "
                          "year regardless of --fix-gas; this is not a recourse leg "
                          "(Section 2, third bullet) -- 'free' gas that cannot be "
                          "built is not free.")}
    return {"passed": True, "fix_gas": fix_gas, "gas_build_case": gas_build_case, "reason": None}


def check_gas_deliverability_baseline(stage1_summary: dict) -> dict:
    case = stage1_summary["standard_arms"].get("gas_deliverability_case")
    passed = case == "baseline"
    return {
        "passed": passed,
        "gas_deliverability_case": case,
        "reason": None if passed else (
            f"Stage-1 gas_deliverability_case={case!r}, not 'baseline' (the mode "
            f"of the Stage-2 sampling distribution, TASK 01 item 18)."),
    }


def check_ws_case_scenario_matches_stage1(stage1_summary: dict, mc_meta: dict, ws_case: str) -> dict:
    """
    TASK 08, Part D. Guard 5 -- ws_case_scenario_matches_stage1. Compares
    the selected --ws-case against 30_'s Stage-1 configuration on every
    field TASK 08 Part B's enumeration found differing: capital_case,
    financing_regime, required_margin, has_emissions_cap. Reads both
    sides from the provenance each script now records for exactly this
    purpose (30_'s stage1_scenario_fields, 17_'s case_fields) -- not
    re-derived or assumed here.
    """
    stage1_fields = stage1_summary.get("stage1_scenario_fields")
    if stage1_fields is None:
        return {
            "passed": False,
            "reason": ("Stage-1 summary has no stage1_scenario_fields -- "
                      "re-run 30_run_stochastic_redispatch.py with the "
                      "current version of the script (TASK 08 Part D "
                      "prerequisite)."),
        }
    case_fields = mc_meta.get("case_fields", {})
    ws_fields = case_fields.get(ws_case)
    if ws_fields is None:
        return {
            "passed": False,
            "reason": (f"17_'s monte_carlo_meta.json has no case_fields entry "
                      f"for --ws-case {ws_case!r} -- re-run 17_run_monte_carlo.py "
                      f"with the current version of the script, or check the "
                      f"--ws-case spelling against {sorted(case_fields)}."),
        }
    diffs = {
        field: {"stage1": stage1_fields[field], "ws_case": ws_fields[field]}
        for field in ("capital_case", "financing_regime", "required_margin", "has_emissions_cap")
        if stage1_fields[field] != ws_fields[field]
    }
    passed = len(diffs) == 0
    return {
        "passed": passed,
        "stage1_scenario_fields": stage1_fields,
        "ws_case_fields": ws_fields,
        "differing_fields": diffs,
        "reason": None if passed else (
            f"--ws-case {ws_case!r} does not match Stage-1 on: "
            f"{', '.join(diffs)}."),
    }


def check_zero_infeasible(stage1_summary: dict, ws_row: dict) -> dict:
    n_infeasible_stage1 = stage1_summary["n_infeasible"]
    n_infeasible_ws = int(ws_row["n_infeasible"])
    passed = (n_infeasible_stage1 == 0) and (n_infeasible_ws == 0)
    return {
        "passed": passed,
        "n_infeasible_stage1": n_infeasible_stage1,
        "n_infeasible_ws": n_infeasible_ws,
        "reason": None if passed else (
            f"{n_infeasible_stage1} infeasible draw(s) in the ECP_free leg, "
            f"{n_infeasible_ws} in the WS leg ({ws_row['case']!r}) -- WS is "
            f"infinite if this count is nonzero for that leg."),
    }


def report_gas_add_banding(stage1_summary: dict) -> dict:
    gas_build_case = stage1_summary["standard_arms"].get("gas_build_case")
    if gas_build_case == "gas_unconstrained":
        return {
            "gas_build_case": gas_build_case,
            "banded": False,
            "detail": ("No band. --fix-gas free applies no constraint of its own "
                      "(30_'s --fix-gas branching has no 'free' clause); "
                      "gas_build_case='gas_unconstrained' leaves gas_add[t] a "
                      "plain NonNegativeReals Var with no finite upper bound. "
                      "RP_partial, if solved, would face the same unbounded "
                      "gas_add -- no band to replicate."),
        }
    elif gas_build_case == "gas_ndc_17gw":
        return {
            "gas_build_case": gas_build_case,
            "banded": True,
            "detail": ("Banded, but NOT as a per-year gas_add band: "
                      "gas_ndc_17gw caps CUMULATIVE gas_capacity_mw[t], "
                      "structurally different from --fix-gas fixed's per-year "
                      "[v*(1-tol), v*(1+tol)] band. RP_partial would need the "
                      "same cumulative cap, not a per-year band."),
        }
    elif gas_build_case == "gas_none":
        return {
            "gas_build_case": gas_build_case,
            "banded": True,
            "detail": "gas_add fixed to 0 for all t -- degenerate, not a recourse leg at all.",
        }
    else:
        return {
            "gas_build_case": gas_build_case,
            "banded": None,
            "detail": f"Unrecognised gas_build_case {gas_build_case!r} -- not analysed by this script.",
        }


def main():
    args = parse_args()

    stage1_summary = json.loads(Path(args.stage1_summary).read_text())
    mc_dir = Path(args.mc_results_dir)
    mc_meta = json.loads((mc_dir / "monte_carlo_meta.json").read_text())
    mc_summary_df = pd.read_csv(mc_dir / "monte_carlo_summary.csv")

    ws_rows = mc_summary_df[mc_summary_df["case"] == args.ws_case]
    if len(ws_rows) == 0:
        print(f"ERROR: --ws-case {args.ws_case!r} not found in "
              f"{mc_dir / 'monte_carlo_summary.csv'}. Available cases: "
              f"{sorted(mc_summary_df['case'].unique())}")
        sys.exit(1)
    ws_row = ws_rows.iloc[0].to_dict()

    guards = {
        "same_draws_sha256": check_same_draws_sha256(stage1_summary, mc_meta),
        "fix_gas_free": check_fix_gas_free(stage1_summary),
        "gas_deliverability_baseline": check_gas_deliverability_baseline(stage1_summary),
        "ws_case_scenario_matches_stage1": check_ws_case_scenario_matches_stage1(
            stage1_summary, mc_meta, args.ws_case
        ),
        "zero_infeasible_both_legs": check_zero_infeasible(stage1_summary, ws_row),
    }
    gas_add_banding = report_gas_add_banding(stage1_summary)
    all_passed = all(g["passed"] for g in guards.values())

    print("\nVSS_partial bound -- guard results")
    print("=" * 70)
    for name, g in guards.items():
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  [{status}] {name}")
        if not g["passed"]:
            print(f"         {g['reason']}")
    print(f"\n  gas_add banding under --fix-gas free "
          f"(gas_build_case={gas_add_banding['gas_build_case']!r}): "
          f"{'BANDED' if gas_add_banding['banded'] else 'UNBOUNDED'}")
    print(f"    {gas_add_banding['detail']}")
    print("\n  Known validity caveats (not gated, always reported):")
    for c in KNOWN_VALIDITY_CAVEATS:
        print(f"    - {c['id']}: {c['detail']}")

    output = {
        "first_stage": "solar_and_storage",
        "ws_case": args.ws_case,
        "guards": guards,
        "all_guards_passed": all_passed,
        "gas_add_banding": gas_add_banding,
        "known_validity_caveats": KNOWN_VALIDITY_CAVEATS,
    }

    if all_passed:
        ecp_free_usd = stage1_summary["mean_cost_usd"]
        ws_usd = float(ws_row["cost_mean"])
        bound_upper_usd = ecp_free_usd - ws_usd
        output.update({
            "bound_emitted": True,
            "ecp_free_usd": ecp_free_usd,
            "ws_usd": ws_usd,
            "bound_upper_usd": bound_upper_usd,
            "bound_upper_pct_of_ecp": (bound_upper_usd / ecp_free_usd * 100
                                      if ecp_free_usd else None),
            "n_draws": stage1_summary["n_draws"],
            "vss_partial_interval": f"vss_partial in [0, {bound_upper_usd:.2f}]",
        })
        output["bound_is_sane"] = bound_upper_usd >= 0
        print(f"\n  ECP_free = ${ecp_free_usd/1e9:.2f}B   WS = ${ws_usd/1e9:.2f}B")
        print(f"  bound_upper = ${bound_upper_usd/1e9:.2f}B "
              f"({output['bound_upper_pct_of_ecp']:.1f}% of ECP_free)")
        print(f"  vss_partial in [0, {bound_upper_usd:.2e}]")
        if bound_upper_usd < 0:
            print("\n  WARNING: bound_upper < 0, i.e. WS > ECP_free. Under a valid "
                  "WS/ECP_free comparison this is impossible (Section 1: WS <= "
                  "RP_partial <= ECP_free). All four named guards passed -- this "
                  "is the known_validity_caveats above (most likely the WS-case "
                  "scenario mismatch, an NDC cap in WS vs none in ECP_free) "
                  "binding in practice, not a guard this script failed to check. "
                  "Do NOT report vss_partial from this run; the four guards are "
                  "necessary, not sufficient, for a valid bound.")
    else:
        output["bound_emitted"] = False
        print("\n  NO BOUND EMITTED -- one or more guards failed (see above).")

    tag = f"stage1={Path(args.stage1_summary).stem}__ws_case={args.ws_case}"
    hash8 = hash8_of(tag)
    append_index(RESULTS_DIR, hash8, tag)
    out_path = RESULTS_DIR / f"vss_bound__{hash8}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
