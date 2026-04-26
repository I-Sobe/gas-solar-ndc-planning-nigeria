"""
14_run_str1_storage_role.py  —  STR-1
======================================

RESEARCH QUESTION (STR-1)
--------------------------
What role does battery storage play in the optimal system — and how does
the deployable hours parameter (H_d) and solar surplus fraction affect
whether storage complements or substitutes for EaaS solar deployment?

THE STORAGE MODEL
-----------------
Storage in this model is an annual energy-shifting device governed by
four constraints (all in optimize_model.py):

1. Charge limited to solar surplus:
   charge[t] <= surplus_frac × solar_generation[t]
   Storage can only charge from solar output — it is coupled to solar.
   There is no grid charging from gas generation.

2. Charge bounded by power capacity (H_d controls throughput):
   charge[t] <= (storage_power_mw[t] × H_d) / 1e6

3. Discharge bounded by round-trip efficiency × charge:
   discharge[t] <= RTE × charge[t]

4. Discharge bounded by power capacity (same H_d):
   discharge[t] <= (storage_power_mw[t] × H_d) / 1e6

Because storage can only charge from solar surplus (Constraint 1), storage
CANNOT EXIST in the optimal plan without solar. A system with zero solar
will build zero storage — they are structurally coupled.

THE TWO PARAMETERS AND WHAT THEY REPRESENT
-------------------------------------------
H_d (storage_deployable_hours_per_year):
  The number of hours per year the battery can charge or discharge at
  rated power. This represents operational infrastructure: inverter sizing,
  grid management capability, dispatch scheduling.

  H_d = 700 (default): ~1.9 h/day average, representing a system that
    dispatches mainly during daily solar peak → deficit transitions.
  H_d = 200: severely constrained operation — typical of early-stage
    BESS with limited dispatch scheduling.
  H_d = 1200: aggressive dispatch — battery can cycle 3+ h/day.

surplus_frac (storage_solar_surplus_frac):
  The fraction of total solar generation available for storage charging.
  This represents the solar-storage integration quality: DC:AC ratio
  design, curtailment policy, co-location arrangement.

  surplus_frac = 0.20 (default): 20% of solar output can charge storage.
  surplus_frac = 0.10: limited integration — most solar goes straight to grid.
  surplus_frac = 0.35: high integration — significant solar-storage coupling.

COMPLEMENT vs SUBSTITUTE
------------------------
COMPLEMENT: Storage and EaaS solar grow together. As H_d increases
  (battery can dispatch more hours), each MW of EaaS solar is more
  valuable (its surplus can be stored and dispatched into evening peaks).
  The model builds MORE EaaS solar when storage is more capable.
  Detected by: solar_eaas_total_mw RISES with H_d.

SUBSTITUTE: Storage replaces marginal EaaS solar. A battery that can
  dispatch for more hours serves demand that would otherwise require
  additional solar MW. The model builds LESS EaaS solar when storage
  is more capable.
  Detected by: solar_eaas_total_mw FALLS with H_d.

The complement_ratio = storage_final_mwh / solar_eaas_total_mw:
  Rising ratio: both scale together (complement)
  Falling ratio: storage grows while EaaS solar shrinks (substitute)

STORAGE UTILISATION
-------------------
storage_utilisation_rate = discharge_total / (capacity × H_d/1e6 × years)
  ≈ 1.0: H_d is the binding constraint — more hours would help
  << 1.0: solar surplus is the binding constraint — more solar (or higher
           surplus_frac) is the bottleneck, not H_d

Binding constraint by year (from storage_binding_by_year):
  power_limit: H_d is binding this year
  charge_limit: solar surplus coupling is binding this year

EXPERIMENT STRUCTURE
--------------------
  H_d values:          [200, 700, 1200]  (low / default / high)
  surplus_frac values: [0.10, 0.20, 0.35] (low / default / high)
  financing_arms:      [public_only, eaas]
  policy_arms:         [no_policy, ndc3_unconditional]

  Total solves: 3 × 3 × 2 × 2 = 36 LP solves

OUTPUTS
--------
  results/str1/str1_results.csv         — all 36 runs
  results/str1/complement_matrix.csv    — complement_ratio by (H_d, surplus_frac)
  results/str1/utilisation_matrix.csv   — storage utilisation and binding constraint
  results/str1/eaas_storage_tradeoff.csv — EaaS solar vs storage by parameter
  results/str1/str1_summary.json        — headline finding: complement or substitute
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io import load_econ
from src.optimize_experiments import run_str1_storage_parameter_sweep

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR    = ROOT / "results" / "str1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAP_PATH       = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"
CANONICAL_VOLL = "voll_mid"
CANONICAL_TARIFF = 95_000_000

# Parameter grids
HD_VALUES          = [200, 700, 1200]
SURPLUS_FRAC_VALUES = [0.10, 0.20, 0.35]

FINANCING_CONFIGS = [
    {
        "label":            "public_only",
        "financing_regime": "traditional",
        "capital_case":     "moderate",
    },
    {
        "label":            "eaas",
        "financing_regime": "eaas",
        "capital_case":     "moderate",
        "required_margin":  1.10,
        "tariff":           CANONICAL_TARIFF,
    },
]

POLICY_CONFIGS = [
    {"policy_label": "no_policy",         "ndc_scenario": None},
    {"policy_label": "ndc3_unconditional", "ndc_scenario": "ndc3_unconditional"},
]


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def compute_complement_matrix(df):
    """
    For each (surplus_frac, financing_arm, policy_label), compute the
    complement_ratio across H_d values and flag whether storage
    complements or substitutes for EaaS solar.

    complement_ratio = storage_final_mwh / solar_eaas_total_mw

    Rising with H_d => complement (storage and EaaS grow together)
    Falling with H_d => substitute (storage displaces EaaS solar)
    """
    rows = []
    for (sfrac, fin, pol), grp in df.groupby(
        ["surplus_frac", "financing_arm", "policy_label"]
    ):
        # Only meaningful for EaaS arm (public_only has no EaaS solar)
        grp_sorted = grp.sort_values("hd")
        ratios     = grp_sorted["complement_ratio"].dropna().values
        hd_vals    = grp_sorted[grp_sorted["complement_ratio"].notna()]["hd"].values

        # Monotonicity of complement_ratio with H_d
        if len(ratios) >= 2:
            diffs    = np.diff(ratios)
            is_rising = bool((diffs >= 0).all())
            is_falling = bool((diffs <= 0).all())
            role = "complement" if is_rising else (
                   "substitute" if is_falling else "mixed")
        else:
            role = None

        for _, r in grp_sorted.iterrows():
            rows.append({
                "surplus_frac":      sfrac,
                "hd":                r["hd"],
                "financing_arm":     fin,
                "policy_label":      pol,
                "storage_final_mwh": r.get("storage_final_mwh"),
                "solar_eaas_mw":     r.get("solar_eaas_total_mw"),
                "complement_ratio":  r.get("complement_ratio"),
                "storage_role":      role,
                "status":            r.get("status"),
            })
    return pd.DataFrame(rows)


def compute_utilisation_matrix(df):
    """
    Show storage utilisation and binding constraint type per (H_d, surplus_frac).
    Identifies whether H_d or solar surplus is the binding operational limit.
    """
    rows = []
    for (hd, sfrac, fin, pol), grp in df.groupby(
        ["hd", "surplus_frac", "financing_arm", "policy_label"]
    ):
        for _, r in grp.iterrows():
            utilisation = r.get("storage_utilisation")
            power_yrs   = r.get("power_limit_years", 0)
            charge_yrs  = r.get("charge_limit_years", 0)

            # Dominant constraint
            if power_yrs > charge_yrs:
                dominant = "H_d (power throughput)"
            elif charge_yrs > power_yrs:
                dominant = "solar_surplus_frac"
            elif power_yrs == 0 and charge_yrs == 0:
                dominant = "neither (slack)"
            else:
                dominant = "tied"

            rows.append({
                "hd":                      hd,
                "surplus_frac":            sfrac,
                "financing_arm":           fin,
                "policy_label":            pol,
                "storage_utilisation":     utilisation,
                "hd_is_binding":           (
                    utilisation is not None and utilisation > 0.90
                ),
                "power_limit_years":       power_yrs,
                "charge_limit_years":      charge_yrs,
                "dominant_constraint":     dominant,
                "status":                  r.get("status"),
            })
    return pd.DataFrame(rows)


def compute_eaas_storage_tradeoff(df):
    """
    For each (H_d, surplus_frac, policy_label), show the EaaS arm:
      - solar_eaas_total_mw at each H_d
      - storage_final_mwh at each H_d
    To make the complement/substitute pattern visible in one table.
    """
    eaas = df[df["financing_arm"] == "eaas"].copy()
    rows = []
    for (pol,), grp in eaas.groupby(["policy_label"]):
        for sfrac, sf_grp in grp.groupby("surplus_frac"):
            for _, r in sf_grp.sort_values("hd").iterrows():
                rows.append({
                    "policy_label":    pol,
                    "surplus_frac":    sfrac,
                    "hd":              r["hd"],
                    "solar_eaas_mw":   r.get("solar_eaas_total_mw"),
                    "storage_mwh":     r.get("storage_final_mwh"),
                    "complement_ratio": r.get("complement_ratio"),
                    "utilisation":     r.get("storage_utilisation"),
                    "dominant":        None,   # filled from utilisation_matrix
                    "status":          r.get("status"),
                })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    STORAGE_OM_CASES = {
        "om_2k":  2_000.0,    # current: LP degeneracy breaker
        "om_10k": 10_000.0,   # NREL ATB low end
        "om_18k": 18_000.0,   # NREL ATB high end
    }

    for om_label, om_value in STORAGE_OM_CASES.items():

        print(f"\n{'='*60}")
        print(f"  STR-1 sweep: storage O&M = {om_label} ({om_value:,.0f} USD/MWh-yr)")
        print(f"{'='*60}")

        econ = load_econ(CANONICAL_VOLL)
        # Override storage O&M for this sweep arm
        econ["STORAGE_OM_PER_MWH_YR"] = om_value

        out_dir = RESULTS_DIR / om_label
        out_dir.mkdir(parents=True, exist_ok=True)

        n_solves = (
            len(HD_VALUES) * len(SURPLUS_FRAC_VALUES)
            * len(FINANCING_CONFIGS) * len(POLICY_CONFIGS)
        )
        print(f"  H_d values:          {HD_VALUES} h/yr")
        print(f"  surplus_frac values: {SURPLUS_FRAC_VALUES}")
        print(f"  financing_arms:      {[f['label'] for f in FINANCING_CONFIGS]}")
        print(f"  policy_arms:         {[p['policy_label'] for p in POLICY_CONFIGS]}")
        print(f"  total solves:        {n_solves}")

        rows = run_str1_storage_parameter_sweep(
            econ=econ,
            hd_values=HD_VALUES,
            surplus_frac_values=SURPLUS_FRAC_VALUES,
            financing_configs=FINANCING_CONFIGS,
            policy_configs=POLICY_CONFIGS,
            cap_path=str(CAP_PATH),
        )

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "str1_results.csv", index=False)
        print(f"\nSaved: {out_dir / 'str1_results.csv'}")

        optimal = df[df["status"] == "optimal"].copy()

        # ── Derived tables ────────────────────────────────────────
        complement_df   = compute_complement_matrix(optimal)
        utilisation_df  = compute_utilisation_matrix(optimal)
        tradeoff_df     = compute_eaas_storage_tradeoff(optimal)

        complement_df.to_csv(out_dir / "complement_matrix.csv",    index=False)
        utilisation_df.to_csv(out_dir / "utilisation_matrix.csv",  index=False)
        tradeoff_df.to_csv(out_dir / "eaas_storage_tradeoff.csv",  index=False)

        # ── Summary JSON ──────────────────────────────────────────
        summary = {"storage_om_usd_per_mwh_yr": om_value,
                    "storage_role_by_policy": {},
                    "parameter_sensitivity": {}}

        for pol in [p["policy_label"] for p in POLICY_CONFIGS]:
            comp_sub = complement_df[
                (complement_df["policy_label"] == pol)
                & (complement_df["financing_arm"] == "eaas")
            ]
            roles = comp_sub["storage_role"].dropna().unique().tolist()
            dominant_role = (
                "complement" if roles.count("complement") > roles.count("substitute")
                else "substitute" if "substitute" in roles
                else "mixed"
            ) if roles else "indeterminate"

            util_sub = utilisation_df[
                (utilisation_df["policy_label"] == pol)
                & (utilisation_df["financing_arm"] == "eaas")
            ]
            hd_binds_fraction = (
                float(util_sub["hd_is_binding"].mean())
                if len(util_sub) else None
            )
            dominant_constraint = (
                util_sub["dominant_constraint"].mode().values[0]
                if len(util_sub) else None
            )

            summary["storage_role_by_policy"][pol] = {
                "dominant_role":       dominant_role,
                "hd_binding_fraction": hd_binds_fraction,
                "dominant_constraint": dominant_constraint,
            }

        # Parameter sensitivity
        eaas_opt = optimal[
            (optimal["financing_arm"] == "eaas")
            & (optimal["policy_label"] == "no_policy")
        ]
        hd_range = (
            float(eaas_opt.groupby("hd")["complement_ratio"].mean().max()
                  - eaas_opt.groupby("hd")["complement_ratio"].mean().min())
            if len(eaas_opt) else None
        )
        sfrac_range = (
            float(eaas_opt.groupby("surplus_frac")["complement_ratio"].mean().max()
                  - eaas_opt.groupby("surplus_frac")["complement_ratio"].mean().min())
            if len(eaas_opt) else None
        )
        summary["parameter_sensitivity"] = {
            "complement_ratio_range_across_hd":          hd_range,
            "complement_ratio_range_across_surplus_frac": sfrac_range,
            "dominant_parameter": (
                "H_d" if (hd_range is not None and sfrac_range is not None
                          and hd_range > sfrac_range)
                else "surplus_frac" if sfrac_range is not None
                else "indeterminate"
            ),
        }

        with open(out_dir / "str1_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # ── Console headline ──────────────────────────────────────
        print(f"\n  === {om_label}: HEADLINE ===")
        for pol, s in summary["storage_role_by_policy"].items():
            print(f"    {pol}: {s['dominant_role'].upper()}")
        print(f"    Dominant parameter: {summary['parameter_sensitivity']['dominant_parameter']}")

    # ── Cross-O&M comparison ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STR-1: Cross-O&M robustness check")
    print(f"{'='*60}")
    for om_label in STORAGE_OM_CASES:
        summary_path = RESULTS_DIR / om_label / "str1_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            om_val = s["storage_om_usd_per_mwh_yr"]
            for pol, role_info in s["storage_role_by_policy"].items():
                print(f"  O&M={om_val:>8,.0f}  {pol:>25s}  → {role_info['dominant_role']}")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
