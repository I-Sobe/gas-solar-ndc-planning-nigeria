# Gas-to-Power Conversion Efficiency Assumption

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
