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
    df = df[df["Year"] >= start_year].sort_values("Year")

    # Gaps INSIDE the source coverage are still an error. Gaps PAST it are the
    # end-effect buffer and are filled by _extend_capex_series.
    found_years = set(int(row["Year"]) for _, row in df.iterrows())
    if not found_years:
        raise ValueError(f"solar_capex.csv has no rows for scenario '{scenario_name}'.")
    last_sourced = max(found_years)
    missing_inside = set(range(start_year, min(end_year, last_sourced) + 1)) - found_years
    if missing_inside:
        raise ValueError(
            f"solar_capex.csv missing years INSIDE its coverage for scenario "
            f"'{scenario_name}': {sorted(missing_inside)}. Check your NREL ATB data."
        )

    result = {}
    for _, row in df.iterrows():
        raw = str(row["Solar_capex_usd_per_mw"]).replace("$", "").replace(",", "").strip()
        result[int(row["Year"])] = float(raw)
    result = {y: v for y, v in result.items() if y <= end_year}
    return _extend_capex_series(result, end_year)


def _extend_capex_series(result, end_year):
    """
    Extend a CAPEX series past its source coverage into the end-effect buffer.

    NREL ATB publishes to 2050; the model runs to scenarios.MODEL_END_YEAR
    (2055) so terminal-year salvage effects fall outside the reporting window.

    Buffer years are HELD FLAT at the final sourced value. Flat is the
    conservative choice: continuing the ATB decline would make buffer-year
    solar progressively cheaper and could pull build forward into the
    reporting window -- precisely the distortion the buffer exists to prevent.
    Buffer-year CAPEX is never reported.
    """
    if not result:
        return result
    last_sourced = max(result)
    if last_sourced >= end_year:
        return result
    terminal = result[last_sourced]
    for y in range(last_sourced + 1, end_year + 1):
        result[y] = terminal
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
    df = df[df["Year"] >= start_year].sort_values("Year")

    # Gaps INSIDE the source coverage are still an error. Gaps PAST it are the
    # end-effect buffer and are filled by _extend_capex_series.
    found_years = set(int(row["Year"]) for _, row in df.iterrows())
    if not found_years:
        raise ValueError(f"storage_capex.csv has no rows for scenario '{scenario_name}'.")
    last_sourced = max(found_years)
    missing_inside = set(range(start_year, min(end_year, last_sourced) + 1)) - found_years
    if missing_inside:
        raise ValueError(
            f"storage_capex.csv missing years INSIDE its coverage for scenario "
            f"'{scenario_name}': {sorted(missing_inside)}."
        )

    result = {}
    for _, row in df.iterrows():
        raw = str(row["Storage_capex_usd_per_mwh"]).replace("$", "").replace(",", "").strip()
        result[int(row["Year"])] = float(raw)
    result = {y: v for y, v in result.items() if y <= end_year}
    return _extend_capex_series(result, end_year)


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

    # SOLAR_CAPEX_PER_MW: 2025 anchor value (solar_low NREL ATB).
    # Used ONLY as fallback when solar_capex_by_year is not passed to build_model().
    # All runner scripts must pass solar_capex_by_year. Do not use this scalar
    # in any post-solve cost diagnostic when time-varying CAPEX is active.
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

    # Thermal-basis emission factor (tCO2 per MWh_th), HHV basis to match the
    # HHV calorific value used in the gas-to-power volume->energy conversion.
    # IPCC default 56.1 kgCO2/GJ (NCV) = 0.2020 tCO2/MWh_th (LHV), converted to
    # HHV basis (/1.108) = 0.1823. EF per MWh_e is DERIVED in build_model as
    # EF_th / gas_eta, so efficiency and emissions can never drift apart.
    # [SOURCE: IPCC default, HHV-adjusted. I'd Replace with Nigeria NIR factor if found.]
    econ["EF_TCO2_PER_MWH_TH"] = 0.1823
    return econ

