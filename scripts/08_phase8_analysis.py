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
        "case": "ndc_unconditional_20",
        "summary_path": PROJECT_ROOT / "results/ndc/ndc_unconditional_20/summary.json",
        "timeseries_path": PROJECT_ROOT / "results/ndc/ndc_unconditional_20/timeseries.csv",
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

def pick(df: pd.DataFrame, names: list[str], case: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"{case}: none of {names} found. Available: {df.columns.tolist()}")

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

    # ---- 8.3 Mechanism decomposition (from timeseries) ----
    decomp_rows = []
    for case, ts in ts_by_case.items():
        # Standard columns expected from your Phase 7C exports
        cols = ts.columns.tolist()
        required = [
            "gas_to_power_twh_th",
            "gas_generation_twh_e",
            "solar_generation_twh_e",
            "storage_discharge_twh_e",
            "unserved_twh",
            "emissions_tco2",
        ]
        for r in required:
            if r not in cols:
                raise ValueError(f"Missing column '{r}' in {case} timeseries.csv")

        decomp_rows.append(
            {
                "case": case,
                "cum_gas_to_power_twh_th": float(safe_numeric(ts["gas_to_power_twh_th"]).sum()),
                "cum_gas_generation_twh_e": float(safe_numeric(ts["gas_generation_twh_e"]).sum()),
                "cum_solar_generation_twh_e": float(safe_numeric(ts["solar_generation_twh_e"]).sum()),
                "cum_storage_discharge_twh_e": float(safe_numeric(ts["storage_discharge_twh_e"]).sum()),
                "cum_unserved_twh": float(safe_numeric(ts["unserved_twh"]).sum()),
                "cum_emissions_tco2_timeseries": float(safe_numeric(ts["emissions_tco2"]).sum()),
            }
        )
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(OUT_DIR / "decomposition_table.csv", index=False)

    # ---- 8.4 Scarcity pricing stats ----
    scarcity_rows = []
    for case, ts in ts_by_case.items():
        # gas shadow
        gas_col = pick(
            ts,
            ["gas_shadow_usd_per_twh_th", "gas_shadow_price_usd_per_twh_th"],
            case,
        )
        g = safe_numeric(ts[gas_col])
        # carbon shadow (baseline may be NaN/0)
        if "carbon_shadow_usd_per_tco2" not in ts.columns:
            raise ValueError(f"Missing 'carbon_shadow_usd_per_tco2' in {case} timeseries.csv")
        c = safe_numeric(ts["carbon_shadow_usd_per_tco2"])

        scarcity_rows.append(
            {
                "case": case,
                "gas_shadow_mean": float(g.dropna().mean()) if g.dropna().shape[0] else np.nan,
                "gas_shadow_max": float(g.dropna().max()) if g.dropna().shape[0] else np.nan,
                "gas_shadow_binding_years": binding_years(g),
                **{f"gas_shadow_{k}": v for k, v in quantiles(g, qs=(0.5, 0.9)).items()},
                "carbon_shadow_mean": float(c.dropna().mean()) if c.dropna().shape[0] else np.nan,
                "carbon_shadow_max": float(c.dropna().max()) if c.dropna().shape[0] else np.nan,
                "carbon_shadow_binding_years": binding_years(c),
                **{f"carbon_shadow_{k}": v for k, v in quantiles(c, qs=(0.5, 0.9)).items()},
            }
        )
    scarcity = pd.DataFrame(scarcity_rows)
    scarcity.to_csv(OUT_DIR / "scarcity_stats.csv", index=False)

    # Plot shadow prices by year
    def plot_shadow(col, fname, title):
        plt.figure()
        for case, ts in ts_by_case.items():
            if col in ts.columns:
                plt.plot(ts["year"].values, safe_numeric(ts[col]).values, marker="o", label=case)
        plt.xlabel("Year")
        plt.ylabel(col)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname, dpi=200)
        plt.close()

    plot_shadow("gas_shadow_usd_per_twh_th", "gas_shadow_by_year.png", "Gas scarcity shadow value by year")
    plot_shadow("carbon_shadow_usd_per_tco2", "carbon_shadow_by_year.png", "Carbon shadow price by year")

    # ---- 8.5 Export vs Power wedge ----
    # Compare power-sector gas scarcity value vs (i) domestic regulated benchmarks and (ii) stylised opportunity regimes

    DOMESTIC_BENCHMARK_USD_PER_MMBTU = {
        "dom_power_2p4": 2.4,
        "dom_commercial_2p9": 2.9,
        "dom_high_3p3": 3.3,  # conservative upper domestic anchor
    }

    OPPORTUNITY_USD_PER_MMBTU = {
        "opp_3": 3.0,
        "opp_5": 5.0,
        "opp_7": 7.0,
    }

    wedge_rows = []
    for case, ts in ts_by_case.items():
        gas_col = pick(
            ts,
            ["gas_shadow_usd_per_twh_th", "gas_shadow_price_usd_per_twh_th"],
            case,
        )
        gas_shadow = safe_numeric(ts[gas_col])

        # Build one iterable of (family, regime_name, usd_per_mmbtu)
        regimes = []
        regimes += [("domestic", name, val) for name, val in DOMESTIC_BENCHMARK_USD_PER_MMBTU.items()]
        regimes += [("opportunity", name, val) for name, val in OPPORTUNITY_USD_PER_MMBTU.items()]

        for family, name, price_mmbtu in regimes:
            price_twh = price_mmbtu * MMBTU_PER_TWH_TH  # USD/TWh_th
            diff = gas_shadow - price_twh  # positive => power value exceeds benchmark

            d = diff.dropna()
            share = float((d > 0).mean()) if d.shape[0] else np.nan

            wedge_rows.append(
                {
                    "case": case,
                    "benchmark_family": family,              # domestic vs opportunity
                    "benchmark_regime": name,                # e.g., dom_power_2p4 / opp_5
                    "benchmark_usd_per_mmbtu": price_mmbtu,
                    "benchmark_usd_per_twh_th": price_twh,
                    "share_years_lambda_power_gt_benchmark": share,
                    "max_gap_usd_per_twh_th": float(d.max()) if d.shape[0] else np.nan,
                    "min_gap_usd_per_twh_th": float(d.min()) if d.shape[0] else np.nan,
                }
            )

    wedge = pd.DataFrame(wedge_rows)
    wedge.to_csv(OUT_DIR / "export_vs_power_wedge.csv", index=False)

    # ---- 8.6 Energy security dashboard (partial: adequacy+sustainability now; reliability/resilience later) ----
    es_rows = []
    for case, ts in ts_by_case.items():
        unserved = safe_numeric(ts["unserved_twh"])
        emissions = safe_numeric(ts["emissions_tco2"])
        carbon_shadow = safe_numeric(ts["carbon_shadow_usd_per_tco2"])

        es_rows.append(
            {
                "case": case,
                "adequacy_cum_unserved_twh": float(unserved.sum()),
                "adequacy_peak_unserved_twh": float(unserved.max()) if unserved.dropna().shape[0] else np.nan,
                "adequacy_years_with_unserved": int((unserved.dropna() > 0).sum()) if unserved.dropna().shape[0] else 0,
                "sustainability_cum_emissions_tco2": float(emissions.sum()),
                "sustainability_binding_years": binding_years(carbon_shadow),
                "sustainability_avg_carbon_shadow_binding_only": float(carbon_shadow[carbon_shadow > 1e-9].mean())
                if (carbon_shadow.dropna() > 1e-9).any()
                else 0.0,
            }
        )
    es = pd.DataFrame(es_rows)
    es.to_csv(OUT_DIR / "energy_security_dashboard.csv", index=False)

    print(f"Phase 8 outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()