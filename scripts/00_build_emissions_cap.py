import json
import os
import pandas as pd
import sys

from pathlib import Path

# Add repo root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


EF_TCO2_PER_TWH_TH = 181100.0  # tCO2 per TWh_th
START_YEAR = 2025
TARGET_YEAR = 2030
END_YEAR = 2045

# Scenarios: 2030 targets defined as % of baseline 2030 (Rule B)
CAPS_2030_MULTIPLIERS = {
    "ndc_unconditional_20": 0.80,  # 20% reduction vs baseline by 2030
    "ndc_conditional_47": 0.53,    # 47% reduction vs baseline by 2030
}

ROOT = Path(__file__).resolve().parents[1]  # repo root
BASELINE_DIAG_PATH = ROOT / "results" / "baseline" / "diagnostics.json"
OUT_PATH = ROOT / "data" / "cost" / "processed" / "emissions_cap.csv"


def main():
    if not BASELINE_DIAG_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline diagnostics at {BASELINE_DIAG_PATH}. "
            f"Run scripts/01_run_baseline.py first."
        )

    with open(BASELINE_DIAG_PATH, "r") as f:
        diag = json.load(f)

    gas_to_power = diag["gas_to_power_twh_th_by_year"]  # dict year -> value

    # Anchor levels
    Ebase_2025 = float(gas_to_power[str(START_YEAR)] if str(START_YEAR) in gas_to_power else gas_to_power[START_YEAR]) * EF_TCO2_PER_TWH_TH
    Ebase_2030 = float(gas_to_power[str(TARGET_YEAR)] if str(TARGET_YEAR) in gas_to_power else gas_to_power[TARGET_YEAR]) * EF_TCO2_PER_TWH_TH

    rows = []
    for scen, mult in CAPS_2030_MULTIPLIERS.items():
        cap2030 = mult * Ebase_2030

        for y in range(START_YEAR, END_YEAR + 1):
            if y <= TARGET_YEAR:
                frac = (y - START_YEAR) / (TARGET_YEAR - START_YEAR) if TARGET_YEAR > START_YEAR else 1.0
                cap_y = (1 - frac) * Ebase_2025 + frac * cap2030
            else:
                cap_y = cap2030

            rows.append({"year": y, "scenario": scen, "cap_tco2": cap_y})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    print("--- Emissions cap written (2030-anchored) ---")
    print("Baseline E2025 (MtCO2):", Ebase_2025 / 1e6)
    print("Baseline E2030 (MtCO2):", Ebase_2030 / 1e6)
    print("Saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
