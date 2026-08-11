import csv
import os
import numpy as np
from src.utils import assert_non_negative


def gas_available_power(start_year, end_year, scenario_name, csv_path=None):
    """
    Load annual gas deliverability to the power sector (TWh_th/year).

    Expected CSV columns:
        year, scenario, gas_available_twh_th

    Returns
    -------
    dict
        {
            "years": np.ndarray,
            "available_twh_th": np.ndarray  # TWh_th/year
        }
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    # Default path: data/gas/processed/gas_available_power_annual_twh_th.csv
    # Adjust if your repo uses a different layout.
    if csv_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(
            repo_root,
            "data",
            "gas",
            "processed",
            "gas_available_power_annual_twh_th.csv",
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Gas availability file not found: {csv_path}")

    years = np.arange(start_year, end_year + 1)

    # Read (year, scenario) -> value
    value_by_year = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"year", "scenario", "gas_available_twh_th"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; found {reader.fieldnames}"
            )

        for row in reader:
            if row["scenario"] != scenario_name:
                continue
            y = int(row["year"])
            v = float(row["gas_available_twh_th"])
            value_by_year[y] = v

    missing = [y for y in years if y not in value_by_year]
    if missing:
        raise ValueError(
            f"Missing gas availability values for scenario='{scenario_name}' years: {missing}"
        )

    avail = np.array([value_by_year[y] for y in years], dtype=float)
    assert_non_negative(avail, "gas availability (TWh_th)")

    return {"years": years, "available_twh_th": avail}

