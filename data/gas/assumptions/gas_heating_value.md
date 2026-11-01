# Gas Heating Value Assumption

> **CORRECTED TASK 73 (2026-10-23). The text below this block is superseded and kept only as the record of the original note.**
>
> - **Basis: HHV (higher heating value), not LHV.** The model's gas-to-power conversion and `gas_eta` are on an HHV basis.
> - **Value: 1,036 Btu/scf (HHV) is an ASSUMED point, not a sourced figure.** It first appears in an uncited working note (`Gas Conversion.docx`, commit `5ec17eb`); no document on disk states it. It sits inside the **sourced range 1,000-1,150 Btu/scf HHV** of the Nigerian Gas Transportation Network Code (DPR edition, issue date UNESTABLISHED, PDF created 2019-11-14; Annex H-1 PDF p.81 and Annex I-1 PDF p.99, reported at 14.73 psia and 60 F). NBSIR 82-2401 Table 10 gives methane 1,014.00 Btu/ft3 at those conditions; a dry ethane-bearing blend comes to about 1,040-1,045. Provenance: `docs/parameter_provenance.md` row 26.
> - **Band carried on `gas_eta`:** at fixed volume, 0.287 x 1036 / H gives **0.2973 at 1,000** to **0.2585 at 1,150** Btu/scf (row 23).
> - **Conversion:** TWh_th = MMscf x 1,036 / 3.412 x 10^6, with 1 MMscf = 1,036 MMBtu at 1,036 Btu/scf.
> - The "~1,030 BTU per cubic foot, Lower Heating Value" below is **wrong on both counts** (no source; labelled LHV while the model is HHV) and is not used by any code.

---

## Superseded original note

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
