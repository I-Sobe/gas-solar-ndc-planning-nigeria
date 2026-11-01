"""
16_plot_cost_decomposition.py  —  System Cost Decomposition Figure
===================================================================

Decomposes NPV system cost into three components:
    - Real expenditure   gas opex + gas capex + solar capex + storage capex
    - Private selfgen    diesel self-generation bought by households/firms
                         when the grid does not serve them (2.6)
    - VoLL penalty       truly unmet demand

CORRECTION PLAN 2.6 CHANGED WHAT THIS FIGURE SHOWS. Before the genset
backstop, unmet grid demand was priced at VoLL and the figure's purpose was to
show that EaaS savings came from avoiding blackout penalties. With a backstop,
unserved is pinned at zero (D11) and the VoLL wedge is structurally absent.

The saving now comes from avoiding PRIVATE DIESEL SPEND -- a real economic
cost borne by Nigerian households and firms, not an assumed penalty parameter.
That is a stronger claim and this figure must make it.

The VoLL wedge is retained so a zero-height bar is visible evidence that the
penalty channel is closed, not silently dropped.

PRE-REQUISITES
--------------
    01_run_baseline.py
    02_run_ndc_caps.py
    03_run_ndc_eaas.py

    All must have been run AFTER the cost_decomposition code was added
    to optimize_experiments.py and optimize_model.py.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_VOLL = "voll_mid"

# ============================================================
# CASES TO PLOT
# ============================================================
# Each entry: (label for chart, path to diagnostics.json)

CASES = [
    {
        "label": "NDC3 Unconditional\n(Public only, tight capital)",
        "path": RESULTS_DIR / "ndc" / f"ndc3_unconditional_{CANONICAL_VOLL}" / "diagnostics.json",
    },
    {
        "label": "NDC3 Unconditional\n(EaaS, tight capital)",
        "path": RESULTS_DIR / "ndc_eaas" / f"ndc3_unconditional_eaas_{CANONICAL_VOLL}" / "diagnostics.json",
    },
    {
        "label": "NDC3 Conditional\n(Public only, moderate capital)",
        "path": RESULTS_DIR / "ndc" / f"ndc3_conditional_{CANONICAL_VOLL}" / "diagnostics.json",
    },
    {
        "label": "NDC3 Conditional\n(EaaS, moderate capital)",
        "path": RESULTS_DIR / "ndc_eaas" / f"ndc3_conditional_eaas_{CANONICAL_VOLL}" / "diagnostics.json",
    },
    {
        "label": "Baseline\n(No policy, unconstrained)",
        "path": RESULTS_DIR / "baseline" / "diagnostics.json",
    },
]


def main():
    labels = []
    real_vals = []
    voll_vals = []
    total_vals = []
    selfgen_vals = []
    salvage_vals = []
    om_vals = []
    obj_vals = []
    recon_gaps = []

    for case in CASES:
        p = case["path"]
        if not p.exists():
            print(f"WARNING: {p} not found — skipping {case['label']}")
            continue

        with open(p, "r") as f:
            diag = json.load(f)

        decomp = diag.get("cost_decomposition", None)
        if decomp is None:
            print(f"WARNING: {p} has no cost_decomposition — was it run after the code update?")
            continue

        real_exp = decomp["real_expenditure_npv"]
        voll_pen = decomp["voll_penalty_npv"]
        # Private self-generation (2.6). Falls back to 0.0 for runs predating
        # the backstop, so old diagnostics still render.
        selfgen = decomp.get("selfgen_npv", diag.get("cost_selfgen_npv", 0.0)) or 0.0
        # Salvage is a CREDIT: system_cost_npv subtracts it, so it must be
        # subtracted here too or the decomposition overstates the objective by
        # the terminal-value credit. Falls back to 0.0 for older diagnostics.
        salvage = decomp.get("salvage_npv", 0.0) or 0.0
        # task_audit_fixes_p3 (A50): O&M (+6.602 bn, +30.9% at central) was
        # added as cost_decomposition's own top-level `om_npv` key
        # (task_wrong_output_repair, Part 1) but never read here -- this
        # reader predates that schema change by nearly a month (see the
        # report for the exact commits). Falls back to 0.0 only for runs
        # from before `om_npv` existed at all, matching the existing
        # fallback style for selfgen/salvage above; the completeness guard
        # below is what actually catches a mismatch of this kind now,
        # regardless of which component is the one missing next time.
        om = decomp.get("om_npv", 0.0) or 0.0
        total = real_exp + selfgen + voll_pen - salvage + om

        # Conversion capital charge (Phase 2.8): checked, not missing.
        # `real_expenditure_npv` above already includes it -- it is folded
        # into `capital_charge_total_npv`, which IS `real_expenditure_npv`'s
        # own capital term (src/optimize_experiments.py, cost_capital_total
        # under 'f1_tranche') -- so no separate top-level field was ever
        # omitted for it the way om_npv was. Itemised only inside
        # decomp["components"] as "conversion_capital_charge_npv", included
        # here only in the reconciliation check below, not as its own bar
        # segment: unlike O&M it was never flagged as needing its own
        # visible line, and it is already correctly counted in the total.
        components = decomp.get("components", {}) or {}
        gas_opex_c = components.get("gas_opex_npv", 0.0) or 0.0
        carbon_c = components.get("carbon_cost_npv", 0.0) or 0.0
        if "capital_charge_total_npv" in components:
            capital_c = components["capital_charge_total_npv"] or 0.0
        else:
            # pre_f1_overnight has no single total key -- sum its itemised
            # capital channels instead (src/optimize_experiments.py's own
            # _capital_components for that branch).
            capital_c = sum(
                components.get(k, 0.0) or 0.0
                for k in ("gas_capex_npv", "public_solar_capex_npv",
                          "eaas_solar_capex_npv", "storage_capex_npv",
                          "replacement_capex_npv")
            )
        real_exp_check = gas_opex_c + capital_c + carbon_c
        recon_gap = (abs(real_exp_check - real_exp) / real_exp
                     if real_exp else 0.0)
        recon_gaps.append((case["label"], recon_gap, real_exp, real_exp_check))

        labels.append(case["label"])
        real_vals.append(real_exp / 1e9)
        selfgen_vals.append(selfgen / 1e9)
        voll_vals.append(voll_pen / 1e9)
        salvage_vals.append(salvage / 1e9)
        om_vals.append(om / 1e9)
        total_vals.append(total / 1e9)
        # Objective from the run's summary.json (not in diagnostics.json), for
        # the reconciliation check below. The decomposition must sum to the
        # solved objective; a gap means a component is missing from cost_total.
        obj = float("nan")
        sp = p.parent / "summary.json"
        if sp.exists():
            with open(sp, "r") as f:
                obj = json.load(f).get("npv_total_cost_usd", float("nan"))
        obj_vals.append(obj / 1e9)

    if not labels:
        print("ERROR: No valid cases found. Check file paths.")
        return

    # ── Plot ───────────────────────────────────────────────────────
    n = len(labels)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked horizontal bars
    # task_audit_fixes_p3 (A50): O&M added as its own segment, same
    # visibility carbon/salvage/selfgen already have in the underlying
    # cost_decomposition -- omitting it from the bar would leave the
    # visual bar length inconsistent with the (now-corrected) printed
    # total. Placed directly after real expenditure: both are public/
    # real resource costs, as distinct from selfgen (private) and VoLL
    # (unserved) that follow.
    bars_real = ax.barh(y_pos, real_vals, height=0.6,
                        color="#2196F3", label="Real expenditure (CAPEX + OPEX)")
    bars_om = ax.barh(y_pos, om_vals, height=0.6,
                      left=real_vals,
                      color="#9C27B0", label="O&M (fixed + variable)")
    left_selfgen = [r + o for r, o in zip(real_vals, om_vals)]
    bars_selfgen = ax.barh(y_pos, selfgen_vals, height=0.6,
                           left=left_selfgen,
                           color="#795548",
                           label="Private diesel self-generation")
    left_voll = [l + s for l, s in zip(left_selfgen, selfgen_vals)]
    bars_voll = ax.barh(y_pos, voll_vals, height=0.6,
                        left=left_voll,
                        color="#FF5722", label="VoLL penalty (truly unserved)")

    for i in range(n):
        total = total_vals[i]
        real_share = real_vals[i] / total * 100 if total > 0 else 0
        om_share = om_vals[i] / total * 100 if total > 0 else 0
        sg_share = selfgen_vals[i] / total * 100 if total > 0 else 0
        ax.text(total + max(total_vals) * 0.01, y_pos[i],
                f"${total:.1f}B\n({real_share:.0f}% capex/opex, {om_share:.0f}% O&M, "
                f"{sg_share:.0f}% private diesel)",
                va="center", ha="left", fontsize=9,
                fontweight="bold" if sg_share > 50 else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("NPV System Cost (USD Billions, 2025 base year)", fontsize=11)
    ax.set_title("System Cost Decomposition: Real Expenditure vs Private Self-Generation vs VoLL Penalty",
                 fontsize=13, fontweight="bold")

    ax.legend(loc="lower right", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))

    # Invert y-axis so first case is on top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save
    fig.savefig(OUT_DIR / "cost_decomposition.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "cost_decomposition.pdf", bbox_inches="tight")
    print(f"\nSaved: {OUT_DIR / 'cost_decomposition.png'}")
    print(f"Saved: {OUT_DIR / 'cost_decomposition.pdf'}")

    # ── Console summary ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  COST DECOMPOSITION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Case':<45} {'Real':>9} {'O&M':>9} {'Selfgen':>9} {'VoLL':>9} {'Salvage':>9} "
          f"{'Total':>10} {'Obj':>10}")
    print(f"  {'-'*85}")
    worst = 0.0
    for i in range(n):
        total = total_vals[i]
        label_oneline = labels[i].replace('\n', ' / ')
        obj = obj_vals[i]
        flag = ""
        if obj == obj and obj > 0:              # NaN-safe equality check
            gap = abs(total - obj) / obj
            worst = max(worst, gap)
            if gap > 1e-6:
                flag = f"  <-- MISMATCH {gap:.2%}"
        print(f"  {label_oneline:<45} {real_vals[i]:>9.1f} {om_vals[i]:>9.1f} "
              f"{selfgen_vals[i]:>9.1f} {voll_vals[i]:>9.1f} {-salvage_vals[i]:>9.1f} "
              f"{total:>10.1f} {obj:>10.1f}{flag}")

    print()
    if worst > 1e-6:
        print(f"  RECONCILIATION FAILED: worst gap {worst:.2%}. The decomposition")
        print("  does not sum to the solved objective, so a component is missing")
        print("  from cost_total in optimize_experiments.py. Do not read shares")
        print("  off this figure until it closes.")
    else:
        print("  Reconciliation OK: real + O&M + selfgen + VoLL - salvage = objective in every case.")

    # ── Completeness guard (task_audit_fixes_p3, A50, Part 1 step 3) ──
    # This is the guard the ORIGINAL bug had none of: it checks
    # real_expenditure_npv's OWN claimed composition (gas opex + capital
    # charge total + carbon, read from decomp["components"]) against the
    # top-level real_expenditure_npv value this script actually plots --
    # independent of, and in addition to, the objective-level check above.
    # A future src/ schema change that adds a new component to one side
    # of this equality without the other (exactly how om_npv went unread
    # for a month) will fail this check loudly, rather than silently
    # producing a plausible-looking wrong figure.
    worst_recon = max((g for _, g, _, _ in recon_gaps), default=0.0)
    print()
    if worst_recon > 1e-6:
        print(f"  {'!'*65}")
        print(f"  COMPLETENESS GUARD FAILED: real_expenditure_npv does not equal")
        print(f"  gas_opex_npv + capital_charge_total_npv + carbon_cost_npv from")
        print(f"  decomp['components'] for at least one case (worst gap "
              f"{worst_recon:.2%}). A component was likely added to or removed")
        print(f"  from real_expenditure_npv's own computation in")
        print(f"  src/optimize_experiments.py without this reader being updated")
        print(f"  to match -- the same failure shape that hid the O&M omission.")
        print(f"  DO NOT USE this figure until the mismatch below is understood:")
        for label, gap, claimed, derived in recon_gaps:
            if gap > 1e-6:
                print(f"    {label.replace(chr(10), ' / ')}: claimed "
                      f"{claimed/1e9:.3f}B, derived from components "
                      f"{derived/1e9:.3f}B, gap {gap:.2%}")
        print(f"  {'!'*65}")
    else:
        print("  Completeness guard OK: real_expenditure_npv matches "
              "gas_opex + capital_charge_total + carbon in every case.")


if __name__ == "__main__":
    main()
