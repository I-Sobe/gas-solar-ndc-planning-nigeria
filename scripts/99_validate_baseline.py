"""
Model validation against observed Nigerian 2024 power sector data.

Cross-checks the baseline LP's first-year (2025) output against
actual 2024 generation statistics from NERC / NBS to confirm
the model is anchored to reality.

EPOCH: current as of HEAD 2a9226b (task_wrong_output_repair, Part 2).
Every OBSERVED_2024 figure below is re-verified against this project's
own sourced documents as of that commit; re-check this stamp if either
this file or the sources it cites move again.

Reference data sources:
    - Demand: NERC 2024 gross generation at the busbar = 37.09 TWh
      (src/scenarios.py header docstring: "[SOURCE: NERC 2024 -- gross
      generation 37,093.70 GWh]"). Metering point: GROSS GENERATION sent
      out, not collected/billed energy -- see the loss-ladder note in the
      same docstring. REPLACES the retired 23.08 TWh collected-energy
      base (task_wrong_output_repair, Part 2; methodology_demand_
      horizon.md Sec 1.1).
    - Gas generation: NERC 2024 gas share of the same gross-generation
      total = 25.62 TWh_e (src/scenarios.py header docstring: "of which
      gas 25.62 TWh"). Same source and metering point as demand above,
      not an independent figure.
    - Solar generation: negligible (< 1 TWh installed utility-scale) --
      no NERC utility-scale solar line item; retained as an independent
      order-of-magnitude estimate, not derived from the demand/gas
      source above. See Part 2 note on OBSERVED_2024["solar_generation_twh"]
      below.
    - unserved_twh: REMOVED (task_wrong_output_repair, Part 2) -- see the
      comment at its former place below.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ---- Observed data (2024 actual, Nigerian grid) ----
OBSERVED_2024 = {
    # task_wrong_output_repair, Part 2: was 23.08 (NBS Q1 2024 annualized
    # SERVED/COLLECTED energy) -- the specifically RETIRED demand base
    # (methodology_demand_horizon.md Sec 1.1): a revenue quantity that
    # understated system scale by roughly 40% against gross generation.
    # Corrected to NERC 2024 gross generation at the busbar -- see module
    # docstring above for the full source and metering-point note.
    "demand_twh":          37.09,
    # task_wrong_output_repair, Part 2: was 25.6 with a literal "(update
    # with actual)" TODO comment -- now updated to the precise, sourced
    # NERC 2024 figure (module docstring above), same source as demand.
    "gas_generation_twh":  25.62,
    # task_wrong_output_repair, Part 2: checked, left UNCHANGED. Not one
    # of the two defects TASK.md named, and no document in this project
    # supersedes it with a different sourced 2024 solar generation
    # figure. NOT the same quantity as the "existing solar 0.88 TWh"
    # baseline-capacity dispatch assumption used elsewhere (e.g.
    # src/scenarios.py's solar_min_build_default() docstring,
    # docs/phase_2_11_minbuild_retirement.md) -- that is a model INPUT
    # (assumed existing capacity's notional output), not an independently
    # observed 2024 generation statistic, and several documents citing it
    # (docs/task_methodology_repair.md line ~195) record that specific
    # backcast arithmetic as itself STALE. Substituting it here would
    # conflate two different quantities, not correct a stale one --
    # closer to inventing a comparison value than sourcing one, so left
    # as the pre-existing rooftop + utility estimate.
    "solar_generation_twh": 0.3,
    # task_wrong_output_repair, Part 2: REMOVED, not re-based. Under any
    # genset case (2.6, the self-generation backstop), `unserved` is near
    # zero BY CONSTRUCTION and is not evidence demand is met --
    # methodology_genself_backstop.md records the reportable quantity as
    # `selfgen + unserved`, the grid SERVICE shortfall. Re-basing on that
    # sum would need a sourced OBSERVED 2024 access-gap/self-generation
    # figure to compare against; the only candidate this project has is
    # NIRP's ~27 TWh self-generation estimate, and register L4 records it
    # as "stipulated, not derived: no methodology, no SFC assumption, no
    # citation," with unresolved unit errors in the memo that tried to
    # rebuild it (docs/tier2_reconciliation.md, D26) -- not a figure this
    # file can honestly cite as an observed comparison value. Per TASK.md
    # Part 2 Sec 3's explicit instruction ("If re-basing needs an
    # observed figure you cannot source, remove it -- do not invent a
    # comparison value"), the check is removed rather than re-based on an
    # invented number.
}

# ---- Load baseline LP 2025 output ----
with open(ROOT / "results/baseline/timeseries.csv", "r") as f:
    ts = pd.read_csv(f)

row_2025 = ts[ts["year"] == 2025].iloc[0]

MODEL_2025 = {
    "demand_twh":          float(row_2025["demand_twh"]),
    "gas_generation_twh":  float(row_2025["gas_generation_twh_e"]),
    "solar_generation_twh": float(row_2025["solar_generation_twh_e"]),
    # task_wrong_output_repair, Part 2: "unserved_twh" key removed along
    # with OBSERVED_2024["unserved_twh"] above -- see that comment. Left
    # unread by the validation loop below (it only iterates
    # OBSERVED_2024's keys), so kept out of this dict rather than left in
    # as a dead, unused entry.
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