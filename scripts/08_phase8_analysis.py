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


