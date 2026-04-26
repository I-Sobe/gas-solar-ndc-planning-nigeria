"""
Model validation against observed Nigerian 2024 power sector data.

Cross-checks the baseline LP's first-year (2025) output against
actual 2024 generation statistics from NERC / NBS to confirm
the model is anchored to reality.

Reference data sources:
    - Demand: NBS Q1 2024 annualized = 23.08 TWh
    - Gas generation: NERC 2024 annual grid generation (~27-30 TWh)
    - Solar generation: negligible (< 1 TWh installed utility-scale)
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ---- Observed data (2024 actual, Nigerian grid) ----
OBSERVED_2024 = {
    "demand_twh":          23.08,   # NBS Q1 2024 annualized served energy
    "gas_generation_twh":  25.6,    # NERC 2024 grid annual gen (update with actual)
    "solar_generation_twh": 0.3,    # rooftop + utility estimate
    "unserved_twh":         3.0,    # from access gap estimates
}

# ---- Load baseline LP 2025 output ----
with open(ROOT / "results/baseline/timeseries.csv", "r") as f:
    ts = pd.read_csv(f)

row_2025 = ts[ts["year"] == 2025].iloc[0]

MODEL_2025 = {
    "demand_twh":          float(row_2025["demand_twh"]),
    "gas_generation_twh":  float(row_2025["gas_generation_twh_e"]),
    "solar_generation_twh": float(row_2025["solar_generation_twh_e"]),
    "unserved_twh":         float(row_2025["unserved_twh"]),
}

# ---- Compute gaps ----
validation = {}
for k in OBSERVED_2024:
    obs = OBSERVED_2024[k]
    mod = MODEL_2025[k]
    gap_abs = mod - obs
    gap_pct = 100 * gap_abs / obs if obs > 0 else None
    validation[k] = {
        "observed_2024": obs,
        "model_2025":    mod,
        "gap_abs_twh":   gap_abs,
        "gap_pct":       gap_pct,
    }

# ---- Write validation table ----
out = ROOT / "results/validation/validation_table.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(validation, f, indent=2)

# ---- Console summary ----
print("=" * 60)
print("  Model validation: 2025 LP output vs 2024 observed")
print("=" * 60)
for k, v in validation.items():
    gap_pct = f"{v['gap_pct']:+.1f}%" if v['gap_pct'] is not None else "n/a"
    print(f"  {k:25s}  obs={v['observed_2024']:6.2f}  "
          f"mod={v['model_2025']:6.2f}  gap={gap_pct}")