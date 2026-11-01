# The four efficiency parameters: one table, and the common basis the model uses

TASK 74 Part B (2026-10-23). Companion to `docs/parameter_provenance.md` rows 23-26 and `docs/task_log_74.md`. "NERC" is NERC (Nigerian Electricity Regulatory Commission) throughout. Cells state what a document says; where none does, the cell says **not established**.

## B.1 The table

| | `gas_eta` (fleet) | `ccgt_eff_iso` / `central` / `low` (new build) | converted stream `conv_low` / `central` / `high` |
|---|---|---|---|
| **Value** | 0.287 (`src/scenarios.py:3214`) | 0.5445 / **0.5173 (central)** / 0.5010 (`:1909-1913`) | 0.42 / 0.46 / 0.50 (`:1399-1411`) |
| **Taxonomy** | DERIVED: numerator SOURCED, denominator and heating value unsourced (row 23) | base SOURCED, derates ASSUMED (row 24) | ASSUMED, no source, a declared family (row 25) |
| **Source / bracket** | NERC 2024 Annual Report (total generation 37,093.70 GWh less hydro 11,469.85 GWh = 25.62 TWh_e, PDF p.26) over a gas volume of about 805 MMscf/d (not in any document on disk) at an assumed 1,036 Btu/scf. Project sweep U(0.277, 0.298); heating-value band **0.2585-0.2973** (0.287 x 1036 / H, H = 1,150 to 1,000 Btu/scf) | EIA / Sargent & Lundy SL-018001 Rev A, Table 5-1, PDF p.76: 6,266 Btu/kWh, 3,412 / 6,266 = 0.5445; x 0.95 (5% derate, ASSUMED) = 0.5173; x 0.92 (8%, ASSUMED) = 0.5010 | The arm's own bracket 0.42-0.50, ASSUMED; no document |
| **Heating value** | HHV by the repository's label; 1,036 Btu/scf is an **ASSUMED point inside the sourced 1,000-1,150 Btu/scf HHV range** (Network Code Annex H-1, PDF p.81) | **HHV, stated** (Table 5-1 row "Net Plant Heat Rate, Higher Heating Value (HHV) Basis"); S&L reference gas HHV 1,040 Btu/scf (Table 1-4, PDF p.26) | HHV by the code's label ("0.46 HHV", `scenarios.py:1370`); no document |
| **Plant gate or net of auxiliary load** | NERC "total generation". Stated (TASK 74): sent-out generation. **Documents:** the NERC reports use "energy generated", "energy sent out" and "energy injected into the grid" in one sentence of the transmission-loss definition (Q4 2025 report, section 2.2.1; 2024 report PDF p.157) as if the same; but the 2024 report's Table D.6 gives "Energy Injected into Grid" 35,711 GWh against total generation 37,094 GWh, **3.7% (1,383 GWh) apart and unreconciled** (PDF p.199). Gross vs net **not established by a document** | **Net of internal auxiliary load, stated** (PDF p.73: CTs about 436 MW gross each, STG about 393 MW gross, "after deducting internal auxiliary power demand, the net output of the plant is 1227 MW"; about 38 MW, 3.0%, my arithmetic) | **Not established** |
| **Ambient conditions** | Realised: whatever Nigerian ambient and site conditions the 2024 fleet ran at | **60 F and 60% RH** (Table 5-1), generic US site; **not named ISO** (ISO appears only in Appendix B). The 5% and 8% Nigerian-ambient derates are ASSUMED and not from S&L | **Not established** (the arm applies "the same 5% ambient derate discipline", `:1372`, a statement in a comment, not a document) |
| **Availability or derating embedded** | **Yes**: realised part load, degradation, cycling, mechanical and fuel outages' effect on efficiency (the ratio is of generation to fuel actually burned). Plant availability itself is separate (`gas_avail_*`) | **No**: a rated heat rate for a new plant; no availability, degradation or cycling in the figure (the report states none). Derates are a separate, ASSUMED step | **Not established**; design-point by construction ("design-point efficiency", `chapter4_audit_pass_3.md:204`) |
| **Fleet average?** | **Yes**: open cycle 59.2%, combined cycle 21.7%, gas steam 19.0% of 10,947 MW (NERC 2024 Tables 4.3, 7.1; my sums) | **No**: one technology class, one reference machine: 2x2x1 H-class, 1,227 MW net (row 24 scope note) | **No**: one retrofit class, three assumed values |
| **Transmission loss embedded?** | **No**: generation is measured at the plant (sent-out or gross); transmission loss is downstream (NERC TLF is 8.21% in 2024, Table D.6) | **No**: plant-gate net | **No** |

## B.2 The common basis the model uses (one sentence, recorded in the `ccgt_efficiency_scenarios()` docstring, `src/scenarios.py:1873-1878`)

All four efficiencies are labelled HHV, but the model has **no established common basis**: the new-build value is documented as net of auxiliary load at 60 F / 60% RH and the fleet value (realised generation over fuel burned at an assumed heating value) and converted value are not established by any document on plant gate or ambient basis, so ratios between them are bounded (below), not point values.

## B.3 The common-basis ratio, as a range

Ratio of new-build to fleet efficiency, arithmetic mine:

| Construction | Low | Central | High |
|---|---|---|---|
| Modelled: `ccgt_eff_central` / `gas_eta`, with `gas_eta` over the heating-value band 0.2585-0.2973 | 0.5173 / 0.2973 = **1.740** | 0.5173 / 0.287 = **1.802** | 0.5173 / 0.2585 = **2.001** |
| Same, over all three new-build cases | 0.5010 / 0.2973 = 1.685 | | 0.5445 / 0.2585 = 2.106 |
| TASK 67's like-for-like (fleet ISO expectation 0.3375, per-type inputs in no document, heating-value-independent as stated) | 0.5173 / 0.3375 = **1.533** | 0.5445 / 0.3375 = **1.613** | |

**Against TASK 67's bound of 1.53-1.80: the documents do not narrow it; carrying the heating-value band widens the upper end to 2.00** (central case; 2.11 across all three cases). What the documents did settle: the new-build base 0.5445 is sourced and its basis known (HHV, net, 60 F / 60% RH); the fleet numerator reproduces; the heating value is an assumed point in a sourced range. What they did not: the fleet gas volume, the fleet's gross/net basis, the 0.3375, and any basis for the converted stream. The lower end 1.53 is a design-basis ratio and is untouched by the heating-value band, which acts on the realised fleet value.
