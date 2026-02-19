# Source Log — Cost Data Layer

This document records the provenance, assumptions, and unit conversions
used to populate the processed cost inputs in `data/cost/processed/`.

All values are expressed in **USD** and aligned to the annual planning
horizon (2025–2045), unless otherwise stated.

---

## 1. Solar CAPEX (utility-scale PV)

### Parameter
- **solar_capex_usd_per_mw**
- Units: USD / MW (installed capacity)

### Source
- National Renewable Energy Laboratory (NREL)
- **Annual Technology Baseline (ATB) Electricity Update 2024**
- File: `data/cost/raw/atb/2024_v3_Workbook.xlsx`
- Documentation: `data/cost/raw/atb/ATB_Electricity_2024.pdf`

### Extraction details
- Sheet: `Solar - Utility PV`
- Metric: Overnight Capital Cost (OCC)
- Original units: USD / kW
- Financial case: **Market**
- Technology maturity: **Mature**
- Year used: **2025**

### Scenario mapping
- Conservative → `solar_high`
- Moderate → `solar_mid`
- Advanced → `solar_low`

### Conversion
- USD / MW = (USD / kW) × 1000

### Notes
- Costs represent EPC + balance-of-system as defined by ATB.
- No explicit transmission or interconnection upgrades included.
- Costs assumed flat over time in initial implementation; learning effects
  may be introduced later as a sensitivity.

---

## 2. Storage CAPEX (utility-scale battery storage)

### Parameter
- **storage_capex_usd_per_mwh**
- Units: USD / MWh (energy capacity)

### Source
- National Renewable Energy Laboratory (NREL)
- **Annual Technology Baseline (ATB) Electricity Update 2024**
- File: `data/cost/raw/atb/2024_v3_Workbook.xlsx`
- Documentation: `data/cost/raw/atb/ATB_Electricity_2024.pdf`

### Extraction details
- Sheet: `Utility-Scale Battery Storage`
- Metric: Overnight Capital Cost (OCC)
- Original units: USD / kW (power-based)
- Financial case: **Market**
- Year used: **2025**

### Assumed configuration
- Battery duration: **4 hours**
- Rationale: ATB default for utility-scale storage;
  aligns with planning-level abstraction used in this study.

### Scenario mapping
- Conservative → `storage_high`
- Moderate → `storage_mid`
- Advanced → `storage_low`

### Conversion
- USD / MWh = (USD / kW) × Duration (hours)
- For 4-hour system: USD / MWh = USD / kW × 4

### Notes
- Model uses energy capacity only (no explicit power constraint).
- O&M and degradation effects are excluded at planning level.

---

## 3. Gas fuel cost (delivered to power sector)

### Parameter
- **gas_cost_usd_per_twh_th**
- Units: USD / TWh_th (thermal energy basis)

### Source (regulatory framework)
- Nigeria Upstream Petroleum Regulatory Commission (NUPRC)
- **Domestic Gas Delivery Regulations (2022)**
- File: `data/cost/raw/gas_price/Domestic_Gas_Delivery_Regulations_2022.pdf`

### Supplementary references
- Petroleum Industry Act (PIA), domestic gas pricing schedules
- Legal and policy commentary used only for interpretation,
  not as primary numeric sources.

### Status
- **Numeric values to be populated**
- Initial implementation will use low / mid / high brackets
  based on USD / MMBtu assumptions.

### Conversion (to be applied)
- 1 TWh_th ≈ 3.412 × 10^6 MMBtu
- USD / TWh_th = (USD / MMBtu) × 3.412 × 10^6

### Notes
- Gas cost represents fuel price only.
- No upstream CAPEX or infrastructure expansion costs included.
- Gas deliverability constraints are modeled separately.

---

## 4. Carbon pricing and emissions factor

### Parameters
- **carbon_price_usd_per_tco2**
- **emission_factor_tco2_per_mwh**

### Source
- International Energy Agency (IEA)
- World Energy Outlook and electricity emissions references
- Stored in: `data/cost/raw/carbon/`

### Extraction details
- Emission factor corresponds to gas-fired electricity generation.
- Units: tCO₂ / MWh_e

### Status
- Carbon price levels defined exogenously (no, moderate, stringent).
- Emission factor value to be finalized and cited explicitly.

### Notes
- Methane leakage is not modeled.
- Carbon cost applies only to gas-generated electricity.

---

## 5. Unserved Energy Penalty (Value of Lost Load proxy)

### Parameter
- **uep_usd_per_twh**
- Units: USD / TWh_e unserved

### Source
- International Value of Lost Load (VOLL) literature
- Stored in: `data/cost/raw/unserved_energy_penalty/`

### Status
- Numeric values to be populated.
- Low / mid / high scenarios will bracket plausible economic damage.

### Conversion
- USD / TWh = (USD / MWh) × 10^6

### Notes
- Penalty represents economic and social cost of unmet demand.
- Used as an energy security proxy, not a market price.

---

## 6. Discount rate

### Parameter
- **real_discount_rate**
- Units: fraction per year

### Source
- To be defined based on financing context and literature.
- Stored in: `data/cost/raw/`

### Status
- Defined for completeness; not yet applied in objective function.

---

## 7. Conversion worksheets

### Purpose
- Track all unit conversions used in processed cost files.

### Location
