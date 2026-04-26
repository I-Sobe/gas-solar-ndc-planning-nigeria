"""
16_plot_cost_decomposition.py  —  System Cost Decomposition Figure
===================================================================

Reads diagnostics.json from each NDC scenario run and produces a
stacked horizontal bar chart decomposing system cost into:
    - Real expenditure (gas opex + gas capex + solar capex + storage capex)
    - VoLL penalty (value-of-lost-load on unserved energy)

This figure is the visual proof that EaaS cost savings come from
avoiding blackout penalties, not from reducing real capital expenditure.

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
        total = real_exp + voll_pen

        labels.append(case["label"])
        real_vals.append(real_exp / 1e9)   # convert to $B
        voll_vals.append(voll_pen / 1e9)
        total_vals.append(total / 1e9)

    if not labels:
        print("ERROR: No valid cases found. Check file paths.")
        return

    # ── Plot ───────────────────────────────────────────────────────
    n = len(labels)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked horizontal bars
    bars_real = ax.barh(y_pos, real_vals, height=0.6,
                        color="#2196F3", label="Real expenditure (CAPEX + OPEX)")
    bars_voll = ax.barh(y_pos, voll_vals, height=0.6,
                        left=real_vals,
                        color="#FF5722", label="VoLL penalty (unserved energy)")

    # Labels on each bar
    for i in range(n):
        total = total_vals[i]
        real_share = real_vals[i] / total * 100 if total > 0 else 0
        voll_share = voll_vals[i] / total * 100 if total > 0 else 0

        # Total cost label at end of bar
        ax.text(total + max(total_vals) * 0.01, y_pos[i],
                f"${total:.1f}B\n({real_share:.0f}% real, {voll_share:.0f}% VoLL)",
                va="center", ha="left", fontsize=9,
                fontweight="bold" if voll_share > 50 else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("NPV System Cost (USD Billions, 2025 base year)", fontsize=11)
    ax.set_title("System Cost Decomposition: Real Expenditure vs VoLL Penalty",
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
    print(f"  {'Case':<45} {'Real ($B)':>10} {'VoLL ($B)':>10} {'VoLL %':>8}")
    print(f"  {'-'*75}")
    for i in range(n):
        total = total_vals[i]
        voll_pct = voll_vals[i] / total * 100 if total > 0 else 0
        label_oneline = labels[i].replace('\n', ' / ')
        print(f"  {label_oneline:<45} {real_vals[i]:>10.1f} {voll_vals[i]:>10.1f} {voll_pct:>7.1f}%")


if __name__ == "__main__":
    main()
