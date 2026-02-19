# Gas Availability and Deliverability Assumption

## Purpose

This document defines the assumptions governing gas availability to the
power sector in the planning optimization model.

Gas is modeled explicitly as a physical resource prior to conversion
to electricity (Phase 1 Gas Accounting).

The model does NOT simulate reservoir behavior field-by-field.
Instead, it represents gas availability as an exogenous annual
deliverability envelope (TWh_th per year).

---

## System Boundary

Gas availability represents:

- Domestic gas allocated to power (DGDO-relevant volumes)
- Physical production capacity
- Infrastructure constraints (pipelines, compression)
- Institutional and regulatory frictions
- Upstream maturity effects

It does NOT represent:
- Total national gas reserves
- LNG export volumes
- Field-level development optimization

---

## Modeling Structure

For each year y:

G_pwr_y + G_curt_y ≤ G_avail_y

Where:

- G_pwr_y   = Gas allocated to power (TWh_th)
- G_curt_y  = Deliverable but unused gas (TWh_th)
- G_avail_y = Maximum deliverable gas to power (TWh_th)

G_avail_y is scenario-based and read from:

data/gas/processed/gas_available_power_annual.csv

---

## Baseline Anchor (2025)

The 2025 value is anchored to:

- NUPRC monthly gas-to-power delivery data (MMSCF)
- Converted to annual volume
- Converted to thermal energy (TWh_th)

This value reflects observed deliverability to the power sector,
not geological reserves.

---

## Deliverability Scenarios

Three deterministic envelopes are defined:

- downside  → declining deliverability
- baseline  → flat or mild decline
- upside    → moderate improvement / stabilization

These scenarios reflect:

- Reservoir maturity
- Development lag
- Infrastructure performance
- Regulatory enforcement
- Domestic gas prioritization

No endogenous gas allocation between export and power is modeled.
Gas considered here is already committed to power use.

---

## Key Assumption

Gas deliverability is treated as:

- Exogenous
- Deterministic
- Binding when scarcity emerges

The model endogenously determines:

- Gas usage within the availability envelope
- Curtailment if unused
- Shadow price of gas scarcity (dual variable)
