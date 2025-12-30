"""
make_data.py
End-to-end data acquisition and preprocessing pipeline
Thesis-grade, reproducible version
"""

import os
import pandas as pd
import requests
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path("data")

CARBON_PRICE_XLSX = BASE_DIR / "stochastic" / "raw" / "carbon_pricing_dashboard.xlsx"

WORLD_BANK_ACCESS_URL = (
    "https://api.worldbank.org/v2/country/NGA/indicator/EG.ELC.ACCS.ZS"
    "?format=json&per_page=1000"
)

OWID_ENERGY_URL = (
    "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
)

# ----------------------------
# UTILITIES
# ----------------------------
def ensure_dirs():
    dirs = [
        BASE_DIR / "demand",
        BASE_DIR / "gas" / "raw",
        BASE_DIR / "solar",
        BASE_DIR / "cost",
        BASE_DIR / "stochastic" / "raw",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def safe_read_csv(url, description):
    try:
        return pd.read_csv(url)
    except Exception as e:
        print(f"[WARNING] Failed to fetch {description}: {e}")
        return None

# ----------------------------
# DEMAND: OWID electricity generation
# ----------------------------
def fetch_owid_demand():
    df = safe_read_csv(OWID_ENERGY_URL, "OWID energy data")
    if df is None:
        return

    nga = df[df["country"] == "Nigeria"]
    out = nga[["year", "electricity_generation"]].dropna()

    out.to_csv(
        BASE_DIR / "demand" / "electricity_generation_owid.csv",
        index=False,
    )

# ----------------------------
# DEMAND: World Bank electricity access
# ----------------------------
def fetch_worldbank_access():
    try:
        response = requests.get(WORLD_BANK_ACCESS_URL, timeout=30)
        response.raise_for_status()
        raw = response.json()[1]
    except Exception as e:
        print(f"[WARNING] World Bank access fetch failed: {e}")
        return

    df = pd.DataFrame(raw)[["date", "value"]].dropna()
    df.columns = ["year", "electricity_access_percent"]

    df.to_csv(
        BASE_DIR / "demand" / "electricity_access_worldbank.csv",
        index=False,
    )

# ----------------------------
# GAS: Manual data note
# ----------------------------
def note_gas_data():
    note = (
        "Gas production data must be downloaded manually from NUPRC:\n"
        "https://www.nuprc.gov.ng/gas-production-status-report/\n\n"
        "Place Excel files in:\n"
        "data/gas/raw/\n"
        "Processing handled in a separate script."
    )

    with open(BASE_DIR / "gas" / "README.txt", "w") as f:
        f.write(note)

# ----------------------------
# STOCHASTIC: Carbon pricing (Excel-based, clean)
# ----------------------------
def process_carbon_prices():
    """
    Reads World Bank Carbon Pricing Dashboard workbook,
    detects header row correctly,
    extracts compliance price data,
    reshapes into long format.
    """

    if not CARBON_PRICE_XLSX.exists():
        print("[WARNING] Carbon pricing workbook not found.")
        print("Expected at:", CARBON_PRICE_XLSX)
        return

    # Read sheet raw
    raw = pd.read_excel(
        CARBON_PRICE_XLSX,
        sheet_name="Compliance_Price",
        header=None,
    )

    # --- Detect header row ---
    header_row = None
    for i in range(10):
        row = raw.iloc[i].astype(str).str.lower()
        if "name of the initiative" in row.values:
            header_row = i
            break

    if header_row is None:
        raise RuntimeError("Could not detect header row in Compliance_Price sheet")

    # Assign headers properly
    raw.columns = raw.iloc[header_row]
    df = raw.iloc[header_row + 1 :].reset_index(drop=True)

    # Rename identifier columns
    df = df.rename(
        columns={
            "Name of the initiative": "initiative",
            "Instrument Type": "instrument",
            "Jurisdiction Covered": "jurisdiction",
            "Region": "region",
            "Income group": "income_group",
            "Start date": "start_year",
        }
    )

    # Identify year columns (numeric years)
    year_cols = [
        c for c in df.columns
        if isinstance(c, (int, float)) and int(c) >= 1990
    ]

    if not year_cols:
        raise RuntimeError("No year columns detected in carbon pricing data")

    id_cols = [
        "initiative",
        "instrument",
        "jurisdiction",
        "region",
        "income_group",
        "start_year",
    ]

    # Ensure required columns exist
    missing = [c for c in id_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Reshape to long format
    df_long = df[id_cols + year_cols].melt(
        id_vars=id_cols,
        var_name="year",
        value_name="carbon_price",
    )

    # Clean
    df_long["year"] = df_long["year"].astype(int)
    df_long["carbon_price"] = pd.to_numeric(
        df_long["carbon_price"], errors="coerce"
    )
    df_long = df_long.dropna(subset=["carbon_price"])

    # Save
    df_long.to_csv(
        BASE_DIR / "stochastic" / "carbon_price_reference.csv",
        index=False,
    )

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    ensure_dirs()
    fetch_owid_demand()
    fetch_worldbank_access()
    note_gas_data()
    process_carbon_prices()
    print("Data pipeline completed successfully.")

if __name__ == "__main__":
    main()
