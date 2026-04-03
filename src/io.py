"""
I/O Utilities

Low-level helpers for loading configuration files and
persisting numerical model outputs.

This module does not perform plotting or reporting.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def load_yaml(filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    filepath : str
        Path to YAML file

    Returns
    -------
    dict
        Parsed YAML contents
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_results(output, filepath):
    """
    Save numerical model results to disk.

    Parameters
    ----------
    output : dict or array-like
        Model results to persist
    filepath : str
        Output file path ('.npz' or '.npy')

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if isinstance(output, dict):
        np.savez(filepath, **output)
    else:
        np.save(filepath, output)


def load_solar_capex_by_year(scenario_name="solar_low", start_year=2025, end_year=2045):
    """
    Returns a dict {year: capex_usd_per_mw} from the NREL solar CAPEX CSV.
    Handles UTF-8 BOM, dollar signs, commas, and trailing spaces.
    """
    solar_df = pd.read_csv(
        ROOT / "data/cost/processed/solar_capex.csv", 
        thousands=",",
        encoding="utf-8-sig" # strips BOM automatically
        )
    df = solar_df[solar_df["Scenario"] == scenario_name].copy()
    df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].sort_values("Year")

    expected_years = set(range(start_year, end_year + 1))
    found_years = set(int(row["Year"]) for _, row in df.iterrows())
    missing = expected_years - found_years
    if missing:
        raise ValueError(
            f"solar_capex.csv missing years for scenario '{scenario_name}': "
            f"{sorted(missing)}. Check your NREL ATB data coverage."
        )

    result = {}
    for _, row in df.iterrows():
        raw = str(row["Solar_capex_usd_per_mw"]).replace("$", "").replace(",", "").strip()
        result[int(row["Year"])] = float(raw)
    return result


def load_storage_capex_by_year(scenario_name="Storage_low", start_year=2025, end_year=2045):
    """
    Returns a dict {year: capex_usd_per_mwh} from the NREL storage CAPEX CSV.
    Handles commas and trailing spaces.
    """
    df = pd.read_csv(
        ROOT / "data/cost/processed/storage_capex.csv",
        thousands=",",
    )
    df = df[df["Scenario"] == scenario_name].copy()
    df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].sort_values("Year")

    expected_years = set(range(start_year, end_year + 1))
    found_years = set(int(row["Year"]) for _, row in df.iterrows())
    missing = expected_years - found_years
    if missing:
        raise ValueError(
            f"storage_capex.csv missing years for scenario '{scenario_name}': "
            f"{sorted(missing)}."
        )

    result = {}
    for _, row in df.iterrows():
        raw = str(row["Storage_capex_usd_per_mwh"]).replace("$", "").replace(",", "").strip()
        result[int(row["Year"])] = float(raw)
    return result


def load_econ(voll_case="voll_low", gas_price_case="gas_low"):
    """
    Parameters
    ----------
    voll_case : str
        One of 'voll_low', 'voll_mid', 'voll_high'.
    gas_price_case : str
        One of 'gas_low', 'gas_mid', 'gas_high'.
        Controls the fuel commodity + transport cost assumption.
        NOTE: gas deliverability volume is controlled separately via
        gas_deliverability_case in load_scenario(). These are independent.
        gas_low (no transport tariff): 7.27 M USD/TWh_th
        gas_mid (domestic tariff):    11.01 M USD/TWh_th
        gas_high (export-parity):     14.87 M USD/TWh_th
    """
    def to_float(x):
        return float(str(x).replace(",", "").replace("$","").strip())

    econ = {}

    gas_df = pd.read_csv(ROOT/"data/cost/processed/gas_cost.csv", thousands=",")
    gas_row = gas_df[gas_df["Scenario"] == gas_price_case].iloc[0]
    econ["GAS_COST_PER_TWH_TH"] = to_float(gas_row["total_usd_per_twh_th"])

    solar_df = pd.read_csv(
        ROOT/"data/cost/processed/solar_capex.csv", 
        thousands=",", 
        encoding="utf-8-sig")
    solar_row = solar_df[
        (solar_df["Scenario"]=="solar_low") &
        (solar_df["Year"]==2025)
    ].iloc[0]

    econ["SOLAR_CAPEX_PER_MW"] = to_float(solar_row["Solar_capex_usd_per_mw"])

    storage_df = pd.read_csv(ROOT/"data/cost/processed/storage_capex.csv", thousands=",")
    storage_row = storage_df[
        (storage_df["Scenario"]=="Storage_low") &
        (storage_df["Year"]==2025)
    ].iloc[0]

    econ["STORAGE_COST_PER_MWH"] = to_float(storage_row["Storage_capex_usd_per_mwh"])

    # Storage annual fixed O&M. Primary purpose: break LP degeneracy in storage
    # sizing (without this, any storage capacity >= ~10 GWh is equally optimal).
    # Value is conservative (NREL ATB range: 10,000–18,000 USD/MWh-yr).
    econ["STORAGE_OM_PER_MWH_YR"] = 2_000.0

    voll_df = pd.read_csv(ROOT/"data/cost/processed/unserved_energy_penalty.csv", thousands=",")
    voll_row = voll_df[
        (voll_df["scenario"]==voll_case) &
        (voll_df["year"]==2025)
    ].iloc[0]

    econ["UNSERVED_ENERGY_PENALTY"] = to_float(voll_row["voll_usd_per_twh"])

    econ["CARBON_EMISSION_FACTOR"] = 0.421

    return econ

