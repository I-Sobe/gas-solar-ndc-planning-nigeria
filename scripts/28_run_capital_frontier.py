"""
28_run_capital_frontier.py  —  Cost-Capital Pareto Frontier (Frontier B)
==========================================================================

task_epsilon_constraint, Part 3. See docs/task_epsilon_constraint.md
for the full ε-constraint argument this runner exists to make; this
docstring states only what this specific script does.

RESEARCH QUESTION
------------------
What does system cost look like as a function of how much public
capital the state can mobilise? This is the thesis's own central
question, expressed as a frontier: a plan can be least-cost and
unfinanceable, and this frontier is where that trade-off becomes a
measured RATE (a dual, USD of system cost per USD of public envelope)
rather than an assertion made once at a point comparison (F-10).

THE ε-CONSTRAINT
------------------
Objective minimised: system cost (`m.system_cost_npv`), unchanged.
ε: the public capital envelope, `scenario["public_solar_budget_npv"]`
   -- set directly, between `load_scenario()` and `build_model_sliced()`,
   bypassing the named `capital_case` arm (confirmed possible without
   editing any file outside this task's whitelist --
   docs/task_epsilon_constraint.md, Part 1F).
Dual: the shadow price of public capital,
   `diag["public_budget_shadow_usd_per_usd"]` -- USD of system cost
   per USD of envelope. **This is this frontier's primary output.**

BUILDER: build_model_sliced (task_epsilon_constraint, Part 6B port)
-----------------------------------------------------------------------
This runner used the ANNUAL builder (`build_model`) at its own
creation (Part 3 of this task). `methodology.md` §2 states the SLICED
builder is the reporting model, and every figure in `docs/findings_
frozen.md` is sliced. The gap was not cosmetic: the annual builder
cannot represent OCGT-to-CCGT conversion at all (raises `ValueError`
on any conversion_case != "conv_none"), and conversion is the second-
largest lever in the model at 5.91 bn (F-6). Ported per `docs/task_
epsilon_constraint.md` Part 6's own inventory: `build_model_sliced`
has an identical call signature to `build_model` (confirmed: this
runner already calls it with 100% keyword arguments), builds `m.
public_budget_constraint` with byte-for-byte identical structure to
the annual builder (always scalar -- `solar_public_add`, `storage_
add`, `eaas_subsidy` are annual investment decisions in BOTH builders,
never slice-indexed), and needs no scenario-dict key this runner does
not already supply (every extra key the sliced builder reads is
already populated by `load_scenario()` via the required standard arms
this runner's full-arm surface already demands). `solve_model` is
builder-agnostic (confirmed: `01b_run_baseline_sliced.py` and `26_run_
f3_identification.py` both use it, unmodified, on sliced models) and
keeps importing from `src.optimize_model` unchanged. This runner
computes no per-year generation share or LCOE field (unlike `18_run_
pareto_frontier.py`), so unlike that file, no direct sliced-unsafe
`m.solar_generation[t]`-style index needed fixing here — confirmed by
re-reading this file's own per-point computations after the port
(`solar_total`, `storage_final`, `cum_unserved`, `cum_selfgen` all
read either annual investment/capacity variables, safe in both
builders, or already-slice-aggregated `diag` fields).

No emissions cap is applied here (`carbon_case="no_policy"`, no
`emissions_cap_by_year` passed to `build_model()`) -- Frontier A
(18_run_pareto_frontier.py) already varies the emissions bound; this
script varies exactly one bound, the public capital envelope, per the
ε-constraint method's own discipline (one bound at a time, never the
objective or a constraint's form).

TWO SEPARATE FRONTIERS, NOT ONE
---------------------------------
Run under BOTH `financing_regime="traditional"` and `"eaas"`
(TASK.md Part 3.3). The gap between them, at matched envelope levels,
is the EaaS mechanism's value expressed as a frontier SHIFT -- a
stronger statement than the single point comparison F-10 makes.

A PREDICTION, RECORDED BEFORE THIS SCRIPT IS RUN
---------------------------------------------------
F-12 finds total solar identified at ~49,487 MW, with the public/EaaS
split bounded below by the public budget and above by bankability. So
as the public envelope tightens, EaaS should absorb displaced public
solar at little cost penalty -- a SHALLOW frontier -- until the
bankability ceiling binds, after which further tightening has nowhere
to go but self-generation and the frontier should STEEPEN SHARPLY. If
that kink appears, the dual at the kink is the marginal value of
public capital. If the frontier is instead straight, the bankability
ceiling is not binding in this configuration -- a different finding,
equally reportable. This prediction is recorded so the result is
checked against an expectation rather than rationalised after; nothing
in this script is tuned toward it.

THE SWEEP RANGE
-----------------
Anchored on the four named capital_case arms (read LIVE from
src/scenarios.py's capital_envelope_scenarios(), not hardcoded here --
see the "why live, not hardcoded" note below), plus points below
`tight` (down to and including zero) to locate any infeasibility edge,
plus extra resolution between `tight` and `moderate` where the
predicted kink is most likely to fall, plus `None` (unconstrained) as
the top endpoint. See CAPITAL_ENVELOPE_FRACTIONS below for the exact
grid and the reasoning for each segment's spacing.

WHETHER AN INFEASIBILITY EDGE EXISTS AT ALL IS NOT ASSUMED
---------------------------------------------------------------
Neither this script nor `18_run_pareto_frontier.py` imposes a grid
service obligation (`max_offgrid_fraction`) -- self-generation is
priced and unbounded (the release valve), while `unserved` is fixed at
zero under any priced genset case (methodology_genself_backstop.md
§1.4), not priced and not capped by this script's own logic either --
so the LP always has a release valve (selfgen, not unserved) and may
remain formally FEASIBLE at every envelope level tested,
including zero. If no infeasible point is found anywhere in this
sweep, that is itself the answer to "where is the infeasibility edge"
(no evidence one exists in the tested range) and must be reported as
such, not assumed away or invented.

OUTPUTS
--------
results/capital_frontier/capital_frontier_raw.csv     -- every point,
    both regimes, status-tagged, never dropped.
results/capital_frontier/capital_frontier_endpoints.json -- per-regime
    endpoint characterisation (Part 3.5 / Frontier A's Part 2.3).
results/capital_frontier/capital_frontier_dominated_points.csv --
    any dominated points found (Part 3.5 / Frontier A's Part 2.4).
results/capital_frontier/capital_frontier_eaas_gap.csv -- per matched
    envelope level, cost(traditional) - cost(eaas): the EaaS value as
    a frontier shift (Part 3.3).
results/capital_frontier/capital_frontier.png/.pdf -- the two
    frontiers plotted together, binding points solid, non-binding
    hollow, absent (unconstrained) marked distinctly (Part 3.4/3.5).

PRE-REQUISITES
----------------
    none beyond the standard arm choices this script requires on the
    command line -- it does not read another runner's output.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ, load_solar_capex_by_year
from src.scenarios import (
    load_scenario, capital_envelope_scenarios, solar_min_build_default,
    MODEL_END_YEAR,
)
from src.optimize_model import solve_model
from src.optimize_model_sliced import build_model_sliced
from src.optimize_experiments import extract_planning_diagnostics
from src.cli_arms import add_standard_arms, standard_arms_kwargs, resolve_results_dir

CANONICAL_VOLL = "voll_mid"
CANONICAL_GENSET = "genset_central"

RESULTS_DIR = ROOT / "results" / "capital_frontier"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

# Both financing regimes as SEPARATE frontiers (TASK.md Part 3.3) --
# not a single sweep with financing_regime as a row tag, but two full
# sweeps, compared afterward.
FINANCING_REGIMES = ["traditional", "eaas"]

# task_epsilon_constraint, Part 3.2: the sweep grid, expressed as
# fractions of B_star (capital_envelope_scenarios()'s own basis, read
# live below -- see "why live" note). Why THESE fractions:
#   0.00 - 0.45 (step 0.05, 10 points): below `tight` (0.50), to locate
#       any infeasibility edge. 0.00 is the literal floor -- the state
#       funds zero public capex/subsidy -- not an arbitrary negative
#       value, which would be nonsensical for a capital envelope.
#   0.50, 0.60, 0.70: `tight`, plus two extra points between `tight`
#       and `moderate` at FINER spacing than elsewhere above `tight` --
#       this is where Part 3.6's predicted kink (the bankability
#       ceiling starting to bind) is most likely to fall, so this
#       region gets the most resolution above the below-tight band.
#   0.85, 0.90, 1.00, 1.10, 1.20: `moderate`, one midpoint, `adequacy`,
#       one midpoint, `expansion` -- coarser spacing, since this region
#       is expected (per the prediction) to be the shallow part of the
#       frontier, not where the interesting structure is.
#   None: `unconstrained` -- the top endpoint, required by TASK.md 3.2.
# 17 finite points + None = 18 points per regime; 2 regimes = 36 solves.
CAPITAL_ENVELOPE_FRACTIONS = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.60, 0.70,
    0.85, 0.90, 1.00, 1.10, 1.20,
    None,
]

# Binding-check tolerance, RELATIVE to the envelope -- same reasoning
# as Frontier A's cap_binding (D1's fix to the interaction classifier:
# an absolute-USD tolerance fails across a sweep spanning zero to
# unconstrained). 1%, same value Frontier A uses, for the same reason
# and for direct comparability between the two frontiers' binding
# conventions.
BUDGET_BINDING_REL_TOL = 0.01


def parse_args():
    # This is a full-arm runner (TASK.md Part 3.1): every standard arm
    # required, no defaults, matching every other repaired runner in
    # this project. capital_case is deliberately NOT one of the
    # standard arms (matches src/cli_arms.py's own design -- capital_case
    # is never a CLI arm; see 01b_run_baseline_sliced.py's own
    # convention) because this script bypasses it entirely -- the
    # public capital envelope IS this script's swept ε, set directly
    # (3.1 below), not chosen from the named arm's fixed values.
    parser = argparse.ArgumentParser(
        description="28_run_capital_frontier: cost-public-capital Pareto frontier (Frontier B)"
    )
    add_standard_arms(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    global RESULTS_DIR
    RESULTS_DIR = resolve_results_dir(RESULTS_DIR, args)   # TASK 53 2.5: arm-keyed output directory
    standard_kwargs = standard_arms_kwargs(args)

    # task_epsilon_constraint, Part 1F / this script's own docstring:
    # read the four named envelope values LIVE from
    # capital_envelope_scenarios(), not hardcoded -- the same lesson
    # this whole repair sequence keeps re-learning (L27 and onward):
    # a second, hand-copied number drifts from its source of truth.
    # src/scenarios.py is read-only here (import, not edit) -- the
    # whitelist forbids EDITING it, not reading from it, exactly as
    # every other runner in this project already does.
    named_envelopes = capital_envelope_scenarios()
    b_star = named_envelopes["adequacy"]  # adequacy == 1.00 * B_star exactly

    print(f"\nCapital Frontier Sweep (Frontier B)")
    print(f"  Named envelope values (USD NPV), read live:")
    for label, value in named_envelopes.items():
        print(f"    {label:<15} {value if value is not None else 'None (unconstrained)'}")
    print(f"  B_star (adequacy): {b_star:,}")
    print(f"  financing_regimes: {FINANCING_REGIMES}")
    print(f"  envelope points:   {len(CAPITAL_ENVELOPE_FRACTIONS)} "
          f"({CAPITAL_ENVELOPE_FRACTIONS})")
    print(f"  total solves:      {len(FINANCING_REGIMES) * len(CAPITAL_ENVELOPE_FRACTIONS)}")

    econ = load_econ(CANONICAL_VOLL, genset_case=CANONICAL_GENSET)

    # Reverse-map each named fraction to its label, for annotating rows
    # that land exactly on a named arm (0.50 -> "tight", etc.) -- so
    # the named arms are directly locatable on the frontier, per
    # TASK.md 3.2's explicit requirement.
    frac_to_label = {
        (v / b_star if v is not None else None): k
        for k, v in named_envelopes.items()
    }

    all_rows = []

    for regime in FINANCING_REGIMES:
        print(f"\n  == financing_regime={regime} ==")

        for frac in CAPITAL_ENVELOPE_FRACTIONS:
            envelope = None if frac is None else int(round(frac * b_star))
            named_label = frac_to_label.get(frac)

            # task_epsilon_constraint, Part 3.1: capital_case is passed
            # as a placeholder ("moderate", load_scenario()'s own
            # default) -- it is overwritten immediately below and never
            # reaches build_model() unmodified. land_case="loose" and
            # solar_build_case="deployment_unconstrained" match 18_'s
            # own choices (18_run_pareto_frontier.py), since this
            # script follows that structure (TASK.md 3.1).
            scenario = load_scenario(
                demand_level_case="served",
                demand_case="organic_central",
                capital_case="moderate",
                solar_build_case="deployment_unconstrained",
                land_case="loose",
                carbon_case="no_policy",
                start_year=2025,
                end_year=MODEL_END_YEAR,
                **standard_kwargs,
            )
            scenario["solar_min_build_mw_per_year"] = solar_min_build_default()
            scenario["financing_regime"] = regime
            # required_margin is NOT set manually here (unlike 18_,
            # which predates dscr_case): load_scenario(dscr_case=...)
            # already writes scenario["required_margin"] from the
            # required dscr_case standard arm (src/scenarios.py:3124,
            # confirmed by reading) -- setting it again here would
            # silently overwrite a value the caller's own --dscr-case
            # flag already determined, the same "second, driftable
            # copy" problem this task's own docstring warns against.
            #
            # task_epsilon_constraint, Part 3.1: the ε itself. Set
            # directly between load_scenario() and build_model(),
            # bypassing capital_case entirely -- exactly as 18_ already
            # mutates the scenario dict post-load_scenario() for
            # financing_regime/required_margin (docs/task_epsilon_
            # constraint.md, Part 1F).
            scenario["public_solar_budget_npv"] = envelope

            years = scenario["years"]
            years_int = [int(y) for y in years]
            solar_capex_tv = load_solar_capex_by_year(
                scenario_name="solar_low",
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )

            envelope_str = f"{envelope:,}" if envelope is not None else "unconstrained"
            label_str = f" [{named_label}]" if named_label else ""

            try:
                # task_epsilon_constraint, Part 6B, Part 2: build_model ->
                # build_model_sliced -- keyword arguments unchanged, per
                # the port inventory (Part 6). Nothing else in this
                # per-point block changes.
                m = build_model_sliced(
                    scenario=scenario,
                    econ=econ,
                    solar_capex_by_year=solar_capex_tv,
                )
                status = solve_model(m, scenario=scenario)

                if not status["optimal"]:
                    all_rows.append({
                        "financing_regime": regime,
                        "envelope_fraction": frac,
                        "envelope_usd": envelope,
                        "named_arm": named_label,
                        "status": "infeasible",
                        "npv_cost": None, "realised_public_spend_pv_usd": None,
                        "public_budget_state": None,
                        "public_budget_dual_usd_per_usd": None,
                        "public_budget_slack_usd": None,
                        "public_budget_tolerance_usd": None,
                        "solar_total_mw": None, "storage_mwh": None,
                        "cum_unserved": None, "cum_selfgen": None,
                    })
                    print(f"    envelope={envelope_str}{label_str} -> INFEASIBLE")
                    continue

                diag = extract_planning_diagnostics(m, scenario, econ)
                total_cost = float(pyo.value(m.system_cost_npv))

                # task_epsilon_constraint, Part 3.5: read the dual that
                # is ALREADY there -- extract_planning_diagnostics()
                # already computes it (src/optimize_experiments.py,
                # confirmed hasattr-guarded); no new extractor built.
                public_budget_dual = diag["public_budget_shadow_usd_per_usd"]
                realised_spend = diag["public_budget_realised_spend_pv_usd"]               # PV, the same basis as the PV budget it is compared with (TASK 53 2.3)
                realised_spend_und = diag["public_budget_realised_spend_undiscounted_usd"]

                # ── Part 3.4: THREE states, not two ──────────────────
                if not hasattr(m, "public_budget_constraint"):
                    # ε was None: the constraint was never built --
                    # structurally ABSENT, not slack (docs/task_epsilon_
                    # constraint.md Part 1F / TASK.md §1).
                    public_budget_state = "absent"
                    budget_slack_usd = None
                    budget_tolerance_usd = None
                else:
                    budget_tolerance_usd = BUDGET_BINDING_REL_TOL * envelope
                    budget_slack_usd = (
                        (envelope - realised_spend)
                        if realised_spend is not None else None
                    )
                    if budget_slack_usd is not None and budget_slack_usd <= budget_tolerance_usd:
                        public_budget_state = "binding"
                    else:
                        public_budget_state = "slack"

                solar_total = sum(
                    float(pyo.value(m.solar_public_add[t]))
                    + float(pyo.value(m.solar_eaas_add[t]))
                    for t in range(len(years))
                )
                storage_final = float(pyo.value(m.storage_capacity_mwh[len(years) - 1]))
                cum_unserved = sum(diag["unserved_twh_by_year"].values())
                cum_selfgen = sum(diag["selfgen_twh_by_year"].values())

                all_rows.append({
                    "financing_regime":  regime,
                    "envelope_fraction": frac,
                    "envelope_usd":      envelope,
                    "named_arm":         named_label,
                    "status":            "optimal",
                    "npv_cost":          total_cost,
                    "realised_public_spend_pv_usd": realised_spend,
                    "realised_public_spend_undiscounted_usd": realised_spend_und,
                    "public_budget_state":            public_budget_state,
                    "public_budget_dual_usd_per_usd":  public_budget_dual,
                    "public_budget_slack_usd":         budget_slack_usd,
                    "public_budget_tolerance_usd":      budget_tolerance_usd,
                    "solar_total_mw":    solar_total,
                    "storage_mwh":       storage_final,
                    "cum_unserved":      cum_unserved,
                    "cum_selfgen":       cum_selfgen,
                })

                dual_str = (
                    f"{public_budget_dual:.4f}" if public_budget_dual is not None else "n/a"
                )
                print(f"    envelope={envelope_str}{label_str} -> "
                      f"cost=${total_cost/1e9:.2f}B  state={public_budget_state}  "
                      f"dual={dual_str}  solar={solar_total/1e3:.1f}GW")

            except Exception as e:
                all_rows.append({
                    "financing_regime": regime,
                    "envelope_fraction": frac,
                    "envelope_usd": envelope,
                    "named_arm": named_label,
                    "status": f"error: {e}",
                    "npv_cost": None, "realised_public_spend_pv_usd": None,
                    "public_budget_state": None,
                    "public_budget_dual_usd_per_usd": None,
                    "public_budget_slack_usd": None,
                    "public_budget_tolerance_usd": None,
                    "solar_total_mw": None, "storage_mwh": None,
                    "cum_unserved": None, "cum_selfgen": None,
                })
                print(f"    envelope={envelope_str}{label_str} -> ERROR: {e}")

    # ── Save raw ──────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "capital_frontier_raw.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'capital_frontier_raw.csv'}")

    optimal = df[df["status"] == "optimal"].copy()

    # Sort key treating unconstrained (None) as +inf -- the loosest
    # possible envelope, for ordering purposes only.
    def _sort_key(frac):
        return float("inf") if frac is None else frac
    optimal["_sort_key"] = optimal["envelope_fraction"].apply(_sort_key)

    # ── task_epsilon_constraint, Part 3.5: endpoint characterisation,
    # per financing regime (same shape as Frontier A's Part 2.3). ─────
    print(f"\n{'='*90}")
    print(f"  FRONTIER B ENDPOINTS (per financing regime)")
    print(f"{'='*90}")
    endpoint_summary = {}
    for regime in FINANCING_REGIMES:
        sub = optimal[optimal["financing_regime"] == regime].sort_values("_sort_key")
        binding_sub = sub[sub["public_budget_state"] == "binding"]
        slack_sub = sub[sub["public_budget_state"] == "slack"]
        absent_sub = sub[sub["public_budget_state"] == "absent"]

        # "The envelope ABOVE which the constraint stops binding": the
        # LARGEST binding envelope fraction found (envelope increases
        # -> looser -> less likely to bind, same direction relationship
        # as Frontier A's cap_fraction).
        frontier_begins_at = (
            float(binding_sub["envelope_fraction"].max())
            if len(binding_sub) and binding_sub["envelope_fraction"].notna().any()
            else None
        )
        # "The envelope BELOW which the problem is infeasible": the
        # smallest FEASIBLE envelope found, and (if any) the largest
        # INFEASIBLE one -- the true edge sits between them.
        infeasible_sub = df[(df["financing_regime"] == regime) & (df["status"] != "optimal")]
        finite_feasible = sub[sub["envelope_fraction"].notna()]
        tightest_feasible_frac = (
            float(finite_feasible["envelope_fraction"].min())
            if len(finite_feasible) else None
        )
        finite_infeasible = infeasible_sub[infeasible_sub["envelope_fraction"].notna()]
        loosest_infeasible_frac = (
            float(finite_infeasible["envelope_fraction"].max())
            if len(finite_infeasible) else None
        )
        no_infeasibility_found = len(infeasible_sub) == 0

        endpoint_summary[regime] = {
            "frontier_begins_at_envelope_fraction": frontier_begins_at,
            "n_binding_points":   int(len(binding_sub)),
            "n_slack_points":     int(len(slack_sub)),
            "n_absent_points":    int(len(absent_sub)),
            "tightest_feasible_envelope_fraction": tightest_feasible_frac,
            "loosest_infeasible_envelope_fraction": loosest_infeasible_frac,
            "no_infeasibility_found_in_swept_range": no_infeasibility_found,
        }
        print(f"  {regime:<15} frontier begins (binding) at envelope_fraction<="
              f"{frontier_begins_at}  "
              f"binding/slack/absent points = {len(binding_sub)}/{len(slack_sub)}/{len(absent_sub)}  "
              f"tightest feasible={tightest_feasible_frac}")
        if no_infeasibility_found:
            print(f"    NOTE: no infeasible point found for {regime} anywhere in "
                  f"this sweep (down to envelope=0) -- see this script's own "
                  f"docstring: the LP has no hard service constraint here, so "
                  f"it may be unconditionally feasible regardless of the "
                  f"public envelope. Not assumed; stated as the swept result.")
        else:
            print(f"    loosest infeasible envelope_fraction found: {loosest_infeasible_frac}")

    # ── Part 3.5: non-dominated check (same shape as Frontier A's
    # Part 2.4). Dominance axis: cost vs REALISED public spend (not the
    # nominal envelope -- a loose envelope may go unused, and the
    # achieved-quantity axis is what should not be dominated, mirroring
    # Frontier A's own use of ACHIEVED emissions, not the nominal cap).
    print(f"\n{'='*90}\n  NON-DOMINATED CHECK\n{'='*90}")
    dominated_rows = []
    for regime in FINANCING_REGIMES:
        sub = optimal[optimal["financing_regime"] == regime].reset_index(drop=True)
        for i in range(len(sub)):
            ci = sub.loc[i, "npv_cost"]
            si = sub.loc[i, "realised_public_spend_pv_usd"]
            if pd.isna(ci) or pd.isna(si):
                continue
            for j in range(len(sub)):
                if i == j:
                    continue
                cj = sub.loc[j, "npv_cost"]
                sj = sub.loc[j, "realised_public_spend_pv_usd"]
                if pd.isna(cj) or pd.isna(sj):
                    continue
                if (cj <= ci and sj <= si) and (cj < ci or sj < si):
                    dominated_rows.append({
                        "financing_regime": regime,
                        "dominated_envelope_fraction": sub.loc[i, "envelope_fraction"],
                        "dominated_by_envelope_fraction": sub.loc[j, "envelope_fraction"],
                        "dominated_cost_usd": ci, "dominated_spend_usd": si,
                        "dominator_cost_usd": cj, "dominator_spend_usd": sj,
                    })
    if dominated_rows:
        dom_df = pd.DataFrame(dominated_rows).drop_duplicates(
            subset=["financing_regime", "dominated_envelope_fraction"]
        )
        print(f"  {len(dom_df)} dominated point(s) found -- a solver "
              f"tolerance issue or a non-binding envelope, diagnostic either way:")
        for _, r in dom_df.iterrows():
            print(f"    {r['financing_regime']}: envelope_fraction="
                  f"{r['dominated_envelope_fraction']} is dominated by "
                  f"envelope_fraction={r['dominated_by_envelope_fraction']} "
                  f"(cost {r['dominated_cost_usd']/1e9:.2f}B -> "
                  f"{r['dominator_cost_usd']/1e9:.2f}B)")
        dom_df.to_csv(RESULTS_DIR / "capital_frontier_dominated_points.csv", index=False)
    else:
        print("  None found -- every reported optimal point is non-dominated.")

    with open(RESULTS_DIR / "capital_frontier_endpoints.json", "w") as f:
        json.dump(endpoint_summary, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'capital_frontier_endpoints.json'}")

    # ── task_epsilon_constraint, Part 3.3: the EaaS gap, per matched
    # envelope level -- the EaaS mechanism's value expressed as a
    # frontier shift. ──────────────────────────────────────────────
    gap_rows = []
    trad = optimal[optimal["financing_regime"] == "traditional"]
    eaas = optimal[optimal["financing_regime"] == "eaas"]
    for frac in CAPITAL_ENVELOPE_FRACTIONS:
        t_row = trad[trad["envelope_fraction"] == frac] if frac is not None \
            else trad[trad["envelope_fraction"].isna()]
        e_row = eaas[eaas["envelope_fraction"] == frac] if frac is not None \
            else eaas[eaas["envelope_fraction"].isna()]
        if len(t_row) == 0 or len(e_row) == 0:
            continue
        t_cost = float(t_row["npv_cost"].iloc[0])
        e_cost = float(e_row["npv_cost"].iloc[0])
        gap_rows.append({
            "envelope_fraction": frac,
            "envelope_usd": t_row["envelope_usd"].iloc[0],
            "named_arm": frac_to_label.get(frac),
            "cost_traditional_usd": t_cost,
            "cost_eaas_usd": e_cost,
            "eaas_value_usd": t_cost - e_cost,
        })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(RESULTS_DIR / "capital_frontier_eaas_gap.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'capital_frontier_eaas_gap.csv'}")
    if len(gap_df):
        print(f"\n{'='*90}\n  EAAS VALUE AS A FRONTIER SHIFT (cost_traditional - cost_eaas)\n{'='*90}")
        for _, r in gap_df.sort_values(
            "envelope_fraction", key=lambda s: s.map(lambda v: v if pd.notna(v) else float("inf"))
        ).iterrows():
            label = f" [{r['named_arm']}]" if r["named_arm"] else ""
            frac_str = f"{r['envelope_fraction']:.2f}" if pd.notna(r["envelope_fraction"]) else "unconstrained"
            print(f"  envelope_fraction={frac_str}{label}  "
                  f"EaaS value = ${r['eaas_value_usd']/1e9:+.3f}B")

    # ── Plot: both frontiers, binding solid, slack hollow, absent
    # marked distinctly (Part 3.4/3.5, same discipline as Frontier A's
    # Part 2.5 -- flag, don't drop). ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    regime_styles = {
        "traditional": ("#FF5722", "o", "Public only (traditional)"),
        "eaas":        ("#2196F3", "s", "EaaS"),
    }
    for regime, (color, marker, legend_label) in regime_styles.items():
        sub = optimal[optimal["financing_regime"] == regime].sort_values("_sort_key")
        binding = sub[sub["public_budget_state"] == "binding"]
        slack = sub[sub["public_budget_state"] == "slack"]
        absent = sub[sub["public_budget_state"] == "absent"]
        # x-axis: realised public spend (USD bn) -- the achieved
        # quantity, plotted the same way Frontier A plots ACHIEVED
        # emissions rather than the nominal cap.
        if len(binding):
            ax.plot(binding["realised_public_spend_pv_usd"] / 1e9, binding["npv_cost"] / 1e9,
                    f"{marker}-", color=color, label=f"{legend_label} (binding)",
                    linewidth=2, markersize=6, alpha=0.9)
        if len(slack):
            ax.scatter(slack["realised_public_spend_pv_usd"] / 1e9, slack["npv_cost"] / 1e9,
                       marker=marker, facecolors="none", edgecolors=color,
                       alpha=0.5, s=40, linewidths=1.2, label=f"{legend_label} (slack)")
        if len(absent):
            ax.scatter(absent["realised_public_spend_pv_usd"] / 1e9, absent["npv_cost"] / 1e9,
                       marker="x", color=color, alpha=0.7, s=60,
                       label=f"{legend_label} (unconstrained)")
    ax.set_xlabel("Realised Public Capital Spend (USD Billions, NPV)")
    ax.set_ylabel("NPV System Cost (USD Billions)")
    ax.set_title("Cost-Capital Frontier\n"
                 "solid = envelope binding | hollow = slack | x = unconstrained (absent)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "capital_frontier.png", dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "capital_frontier.pdf", bbox_inches="tight")
    plt.show()

    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
