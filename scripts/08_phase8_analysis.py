import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(".")
OUT_DIR = PROJECT_ROOT / "results_phase8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        "case": "baseline_no_policy",
        "summary_path": PROJECT_ROOT / "results/baseline/summary.json",
        "timeseries_path": PROJECT_ROOT / "results/baseline/timeseries.csv",
    },
    {
        "case": "ndc_conditional_20",
        "summary_path": PROJECT_ROOT / "results/ndc/ndc_conditional_20/summary.json",
        "timeseries_path": PROJECT_ROOT / "results/ndc/ndc_conditional_20/timeseries.csv",
    },
    {
        "case": "ndc_conditional_47",
        "summary_path": PROJECT_ROOT / "results/ndc/ndc_conditional_47/summary.json",
        "timeseries_path": PROJECT_ROOT / "results/ndc/ndc_conditional_47/timeseries.csv",
    },
]

# Netback regimes in USD/MMBtu (scenario anchors, not point estimates)
NETBACK_USD_PER_MMBTU = {
    "low_5": 5.0,
    "mid_8": 8.0,
    "high_12": 12.0,
}

# Conversion:
# 1 MMBtu = 293.071 kWh_th
# 1 TWh_th = 1e9 kWh_th -> 1e9/293.071 MMBtu
MMBTU_PER_TWH_TH = 1e9 / 293.071  # ≈ 3,412,141.6


# =========================
# HELPERS
# =========================
def load_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError(f"'year' column missing in {path}")
    df["year"] = df["year"].astype(int)
    return df


def safe_numeric(series: pd.Series) -> pd.Series:
    # Some runs may output None; pandas reads as NaN automatically.
    return pd.to_numeric(series, errors="coerce")


def quantiles(series: pd.Series, qs=(0.5, 0.9)) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {f"q{int(q*100)}": np.nan for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q*100)}"] = float(s.quantile(q))
    return out


def binding_years(series: pd.Series, tol=1e-9) -> int:
    s = series.dropna()
    return int((s > tol).sum())


# =========================
# PHASE 8 PIPELINE
# =========================
def main():
    # ---- 8.1 Master Results Table ----
    master_rows = []
    ts_by_case = {}
    summary_by_case = {}

    for c in CASES:
        case = c["case"]
        summ = load_summary(c["summary_path"])
        ts = load_timeseries(c["timeseries_path"])

        summary_by_case[case] = summ
        ts_by_case[case] = ts

        dv = summ.get("decision_variables", {})
        master_rows.append(
            {
                "case": case,
                "npv_total_cost_usd": summ.get("npv_total_cost_usd", np.nan),
                "cumulative_emissions_tco2": summ.get("actual_emissions_tco2_total", np.nan),
                "cumulative_unserved_twh": summ.get("cumulative_unserved_twh", np.nan),
                "solar_total_built_mw": dv.get("solar_total_built_mw", np.nan),
                "storage_capacity_mwh": dv.get("storage_capacity_mwh", np.nan),
            }
        )

    master = pd.DataFrame(master_rows)
    master.to_csv(OUT_DIR / "master_results.csv", index=False)

    # ---- 8.2 Frontier + MAC ----
    frontier = master[["case", "npv_total_cost_usd", "cumulative_emissions_tco2"]].copy()
    frontier = frontier.sort_values("cumulative_emissions_tco2", ascending=False)
    frontier.to_csv(OUT_DIR / "frontier_points.csv", index=False)

    # Compute incremental MAC between adjacent points (ordered by emissions)
    mac_rows = []
    for i in range(len(frontier) - 1):
        a = frontier.iloc[i]
        b = frontier.iloc[i + 1]
        d_cost = float(b["npv_total_cost_usd"] - a["npv_total_cost_usd"])
        d_em = float(a["cumulative_emissions_tco2"] - b["cumulative_emissions_tco2"])
        mac = np.nan if d_em == 0 else d_cost / d_em
        mac_rows.append(
            {
                "from_case": a["case"],
                "to_case": b["case"],
                "delta_cost_usd": d_cost,
                "delta_emissions_tco2": d_em,
                "implied_mac_usd_per_tco2": mac,
            }
        )
    mac_df = pd.DataFrame(mac_rows)
    mac_df.to_csv(OUT_DIR / "marginal_abatement_cost.csv", index=False)

    # Frontier plot
    plt.figure()
    plt.plot(
        frontier["cumulative_emissions_tco2"].values,
        frontier["npv_total_cost_usd"].values,
        marker="o",
    )
    for _, r in frontier.iterrows():
        plt.annotate(r["case"], (r["cumulative_emissions_tco2"], r["npv_total_cost_usd"]))
    plt.xlabel("Cumulative emissions (tCO2)")
    plt.ylabel("NPV total system cost (USD)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "frontier_cost_vs_emissions.png", dpi=200)
    plt.close()

    