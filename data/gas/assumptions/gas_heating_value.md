# Gas Heating Value Assumption

## Purpose

This document defines the thermal conversion assumptions used to
convert volumetric gas data into energy units.

Gas is internally modeled in thermal energy terms (TWh_th).

---

## Unit Conversions

Primary source data:

- NUPRC gas-to-power volumes (MMSCF)

Conversion pathway:

1 MMSCF ≈ 1,000,000 cubic feet

Assumed Lower Heating Value (LHV):

≈ 1,030 BTU per cubic foot

Thus:

1 MMSCF ≈ 1.03 × 10^9 BTU
1 MMBtu = 10^6 BTU

So:

1 MMSCF ≈ 1,030 MMBtu

Energy scaling:

1 TWh = 3.412 × 10^6 MMBtu

Therefore:

TWh_th = (MMSCF × 1,030) / (3.412 × 10^6)

---

## Assumption Justification

- Nigerian pipeline gas composition is assumed sufficiently
  consistent to allow a representative heating value.
- Variations in gas composition are not modeled explicitly.
- Thermal basis is used consistently throughout the model.

---

## Modeling Implication

Gas availability is represented in TWh_th.

Electricity conversion occurs via a constant fleet efficiency:

E_gas_y = η × G_pwr_y
