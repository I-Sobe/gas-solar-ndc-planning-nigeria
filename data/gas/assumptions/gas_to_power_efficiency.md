# Gas-to-Power Conversion Efficiency Assumption

> **CORRECTED TASK 73 (2026-10-23). The text below this block is superseded and kept only as the record of the original note.**
>
> - **The model's fleet efficiency is `gas_eta` = 0.287, not 0.43** (`src/scenarios.py`, `"gas_eta": 0.287`; provenance `docs/parameter_provenance.md` row 23). It is a realised whole-fleet ratio: NERC (Nigerian Electricity Regulatory Commission) 2024 generation less hydro, 25.62 TWh_e (37,093.70 - 11,469.85 GWh; *2024 Annual Report*, PDF p.26), over gas-to-power volume at an assumed 1,036 Btu/scf HHV. The thermal-input denominator (about 805 MMscf/d) is not reproducible from any document on disk. Band from the heating-value range: 0.2585-0.2973.
> - **The 0.43 is an artefact**: `data/gas/processed/gas_to_power_reconciliation_2025H1.xlsx` computes it from six months of volume against thermal generation inferred from a partial-period table (register L125, `docs/task_log_67.md`). It is not used by the model.
> - **Efficiency families in the model** (all HHV): fleet `gas_eta` 0.287; new-build combined cycle `ccgt_eff_iso` / `central` / `low` = 0.5445 / 0.5173 / 0.5010 (EIA / Sargent & Lundy SL-018001 Table 5-1, net HHV at 60 F / 60% RH; derates ASSUMED; rows 24); converted stream, a declared family of ASSUMED values **0.42 / 0.46 / 0.50** (`conv_low` / `conv_central` / `conv_high`, no source; row 25).

---

## Superseded original note

## Purpose

This document defines the thermal-to-electric conversion efficiency
used in the model.

Gas is converted from thermal energy (TWh_th) to electricity (TWh_e).

---

## Empirical Derivation

Fleet-average efficiency was derived from:

- NUPRC gas-to-power delivery data (thermal input)
- NERC thermal electricity generation data (electric output)

Reconciliation yields:

η ≈ 0.43

Where:

η = Electricity output / Thermal gas input

---

## Interpretation

η = 0.43 implies:

43% of thermal energy in gas is converted into electricity.

This is consistent with:

- Combined Cycle Gas Turbine (CCGT) systems
- Mixed fleet including OCGT and partial-load inefficiencies

---

## Modeling Equation

For each year y:

E_gas_y = η × G_pwr_y

Where:

- E_gas_y  = Electricity generated from gas (TWh_e)
- G_pwr_y  = Gas allocated to power (TWh_th)
- η        = 0.43 (constant)

---

## Simplifications

- Efficiency is held constant over 2025–2045.
- No degradation or technology improvement modeled.
- Load-factor effects not explicitly modeled.

This assumption is appropriate for planning-level abstraction.
