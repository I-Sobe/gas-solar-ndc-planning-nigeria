# data/demand — Nigeria Electricity Demand (Planning-Level, Scenario-Based)

## Purpose

This folder documents and stores the data used to construct **annual national electricity demand inputs (TWh/year)** for the Nigeria gas–solar planning optimization model.

The model is **planning-level** (annual time step). Demand is treated as an **exogenous scenario input** (magnitude + growth), not econometrically forecast from historical consumption time series.

---

## Core Demand Framing (Examiner-Critical)

Nigeria’s reported electricity consumption is widely understood to be **supply-constrained** (generation availability, fuel constraints, network limitations). Therefore, official statistics are treated as **served demand**—a lower bound on unconstrained demand.

We distinguish:

- **Served demand** \(E_{\text{served}}\): observed electricity energy delivered/billed under constraints.
- **Latent (true) demand** \(E_{\text{true}}\): electricity demand that would be realized under reliable, unconstrained supply.

Latent demand is not directly observed and is reconstructed using bounded assumptions.

---

## Base-Year Anchor (2024)

### Source (served energy)
Base-year magnitude is anchored to the most recent available official statistics:

- **National Bureau of Statistics (NBS) Electricity Report, Q1 2024**
- Reported served energy for Q1 2024:

\[
E_{Q1,2024} = 5{,}770\ \text{GWh} = 5.770\ \text{TWh}
\]

This value is recorded in `nbs_q1_2024_served_energy.csv`.

### Annualization (served demand)
The planning model requires annual energy demand (TWh/year). Since the anchor is quarterly, it is annualized using a first-order scaling:

\[
E_{\text{served},2024} = 4 \times E_{Q1,2024}
= 4 \times 5.770
= 23.08\ \text{TWh/year}
\]

This annualization anchors demand **magnitude** only; it does not attempt to represent seasonality.

---

## Latent Demand Reconstruction (Suppression Factor)

Because served demand is supply-constrained, latent demand is reconstructed via a suppression factor \(\lambda\):

\[
E_{\text{true}} = \frac{E_{\text{served}}}{\lambda}
\]

A bounded envelope is adopted:

- \(\lambda = 0.60\)  → **latent_low** (moderate suppression; lower latent demand)
- \(\lambda = 0.30\)  → **latent_high** (severe suppression; higher latent demand)

Using \(E_{\text{served},2024} = 23.08\ \text{TWh/year}\):

\[
E_{\text{latent,low}} = \frac{23.08}{0.60} = 38.47\ \text{TWh/year}
\]
\[
E_{\text{latent,high}} = \frac{23.08}{0.30} = 76.93\ \text{TWh/year}
\]

These values are stored in `demand_base_annualized_2024.csv`.

---

## Demand-Level Scenarios (Model Input Contract)

The scenario layer consumes base-year demand as `scenario["base_demand_twh"]` and now supports three demand-level cases:

| demand_level_case | Interpretation | base_demand_twh (TWh/year) |
|---|---|---:|
| `served` | Observed served demand (annualized) | 23.08 |
| `latent_low` | Reconstructed latent demand (λ = 0.60) | 38.47 |
| `latent_high` | Reconstructed latent demand (λ = 0.30) | 76.93 |

These are defined in `src/scenarios.py` via `demand_level_scenarios()` and are intended to be traceable to this folder.

---

## Demand Growth Scenarios

Demand growth is applied exogenously using `scenario["demand_growth"]` and `project_baseline_demand()` in `src/demand.py`.

Growth scenarios (fractions per year) are defined in `src/scenarios.py`:

- `low` = 0.025
- `baseline` = 0.04
- `high` = 0.06

The model then projects:

\[
E(t) = E_0(1+g)^{(t-t_0)}
\]

where \(E_0\) is one of the demand-level base values above and \(g\) is the selected growth rate.

---

## Expected Files in This Folder

### Required (minimum, non-negotiable)
- `nbs_q1_2024_served_energy.csv`  
  Q1 2024 served energy anchor (GWh), with source document reference and notes.

- `demand_base_annualized_2024.csv`  
  Annualized served demand (TWh/year) + latent-demand bounds (TWh/year) with λ values.

### Optional (validation / narrative support only)
- `worldbank_kwh_per_capita.csv`  
  For cross-validation and contextual discussion (not a primary demand input).

- `worldbank_population.csv`  
  For scaling checks and contextual discussion.

- `electricity_generation_owid.xlsx`  
  Trend/context check only, not a primary demand input.

---

## Limitations (Explicit)

1. **Quarter-to-annual scaling:** The base-year served-demand anchor uses Q1 2024 and applies a simple 4× annualization. This ignores intra-annual seasonality and operational variability.

2. **Demand is scenario-based:** Due to supply constraints, historical consumption is not demand-revealing; therefore, the model does not fit an econometric demand forecast. Demand uncertainty is handled through bounded scenarios (served vs latent bounds and growth-rate cases).

3. **Annual resolution:** The planning model operates at annual resolution and does not reconstruct hourly load shapes in this thesis.

These limitations are acceptable because the study is a long-term planning optimization and interprets outputs comparatively across scenarios rather than as precise point forecasts.

---

## Provenance and Reproducibility Rules

For each dataset stored here:
- Preserve units and conversion steps (GWh → TWh).
- Record source title, table/page reference, URL, and access date.
- Keep any manual extraction notes (e.g., where “5,770 GWh” appears in the report).

This ensures that all demand inputs are fully traceable and reproducible for examination.
