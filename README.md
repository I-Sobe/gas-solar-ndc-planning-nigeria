# README.md
Energy-as-a-Service Financing for Gas-Constrained Power Systems: Optimising Nigeria's NDC 3.0 Investment Pathway.

## 🔍 Project Overview
This repository contains the full modeling framework, datasets, scripts, and reproducibility instructions for the MSc thesis:

"Energy-as-a-Service Financing for Gas-Constrained Power Systems: 
Optimising Nigeria's NDC 3.0 Investment Pathway"

This repository contains the full modelling framework, datasets, and 
run scripts for an MSc thesis that builds a deterministic, multi-scenario 
linear programming model of Nigeria's on-grid utility-scale power sector 
over the 2025–2045 planning horizon. The model minimises total system NPV 
cost (comprising real capital/operating expenditure and value-of-lost-load 
penalties) subject to gas deliverability constraints, NDC 3.0 annual 
emissions caps (derived via proportional top-down apportionment of the 
energy-sector abatement target), a public capital budget ceiling covering 
solar and storage CAPEX, EaaS bankability constraints, and a solar build 
rate cap. It evaluates whether Energy-as-a-Service private financing 
can close the investment gap that prevents simultaneous supply adequacy 
and climate compliance under Nigeria's gas-constrained, capital-constrained 
baseline.
This work supports insights for:
* National planners and regulators (MoP/NERC)
* Utilities and generation companies
* Renewable IPPs
* EaaS providers and commercial/industrial customers
* International development partners (WB, AfDB, IRENA)

## Modeling Philosophy and Architectural Mapping

### Modeling Objective
This repository implements a **decision-oriented techno-economic optimization model** to evaluate Nigeria’s long-term electricity generation mix under **delivered gas-to-power availability constraints**, renewable deployment frictions, financing structures, and policy uncertainty.

The model is designed for **national-scale, long-term planning**, focusing on system-level outcomes:
- total system cost,
- electricity reliability (unserved energy),
- carbon emissions.

It is not intended to simulate individual reservoirs, gas fields, or detailed power-flow dynamics.


### What Is Modeled Endogenously
The following elements are **endogenous decision variables or system outcomes** within the optimization framework:

- Installed capacity of:
  - utility-scale solar photovoltaic generation (public and EaaS-financed),
  - battery energy storage.
- Annual generation and dispatch subject to availability and technical limits.
- Unserved energy as an explicit reliability metric.
- System cost, emissions, and reliability trade-offs.
- Financing-dependent deployment feasibility (e.g., utility CAPEX vs Energy-as-a-Service).


### What Is Scenario-Based (Exogenous by Design)
The following elements are represented through **structured scenarios and parameters**, not endogenously optimized:

- **Delivered gas-to-power availability**, defined as the maximum annual electricity that can be generated from gas-fired plants.
  - This availability is informed by:
    - upstream deliverability trends (including decline analytics),
    - midstream infrastructure constraints,
    - domestic gas allocation and pricing regimes.
- Gas-price and carbon-price trajectories.
- Domestic gas policy regimes (allocation priorities, pricing structures).
- Renewable regulatory environments (grid-integration limits, tariff and PPA feasibility).
- Financing environments (WACC, investment ceilings, deployment rates).

This separation reflects the resolution of publicly available data and the scope of national-scale planning studies.


### Interpretation of Gas Decline
Gas decline in this model **does not represent reservoir simulation or field-level forecasting**.

Instead:
- Decline analytics are used to inform **aggregate deliverability trends**.
- These trends contribute to **delivered gas-to-power availability envelopes**, which act as binding constraints in the electricity system optimization.

The system therefore responds to **delivered gas constraints**, not to subsurface production mechanics.


### Representation of Energy-as-a-Service (EaaS)
Energy-as-a-Service (EaaS) is modeled **mechanistically**, not narratively.

EaaS affects system outcomes through one or more of the following channels:
- reduced effective cost of capital,
- higher feasible deployment rates,
- alternative CAPEX–OPEX structures,
- relaxed affordability or investment constraints.

EaaS is treated as a **deployment and financing enabler**, not a separate generation technology.


### Reliability Treatment
Electricity reliability is represented through **unserved energy**, calculated annually and penalised economically in the objective function at $10,000/MWh (VoLL). An explicit reliability constraint (maximum unserved fraction epsilon) provides an optional hard backstop. System cost decomposes into real expenditure (CAPEX + OPEX) and VoLL penalty, allowing reliability to be:
- explicitly traded against cost and emissions,
- evaluated across gas-availability, policy, and financing scenarios,
- decomposed into capital-driven and blackout-driven cost components.

### Architectural Mapping: Text → Code
Key modeling concepts map directly to repository modules:

| Concept | Module |
|------|------|
| Gas-to-power availability | `src/gas_supply.py` |
| Policy and availability scenarios | `src/scenarios.py` |
| Demand projection | `src/demand.py` |
| Solar generation | `src/solar.py` |
| Battery storage | `src/storage.py` |
| Ex-post feasibility dispatch | `src/dispatch.py` |
| Cost, emissions, financing | `src/economics.py` |
| Optimization formulation | `src/optimize_model.py` |
| Experiment orchestration | `src/optimize_experiments.py` |
| Visualization and Pareto analysis | `notebooks/visualization.ipynb` |
| Cost decomposition & figures | `scripts/16_plot_cost_decomposition.py` |
| Validation | `scripts/99_validate_baseline.py` |
| I/O and serialisation | `src/io.py`, `src/utils.py` |

This mapping ensures traceability between methodological claims in the thesis and executable model components.

### Intended Use and Limitations
The model is intended for:
- comparative scenario analysis,
- policy and planning insight,
- evaluation of robustness under uncertainty.

It is not intended for:
- reservoir engineering studies,
- nodal transmission analysis,
- short-term operational dispatch.

These limitations are deliberate and consistent with the research objectives.

## 🧠 Core Contributions

This thesis addresses one core research question: under binding gas 
deliverability constraints, capital scarcity, and Nigeria's NDC 3.0 
emissions targets (derived via proportional top-down apportionment of 
the energy-sector abatement target to gas-fired power generation), can 
Energy-as-a-Service financing resolve the investment gap that prevents 
simultaneous supply adequacy and climate compliance — and at what cost?

Thirteen sub-questions are structured across the four dimensions of the 
World Energy Council's 4As energy security framework (Availability, 
Accessibility, Affordability, Acceptability), ensuring that model 
outputs map directly to recognised policy-relevant criteria rather 
than to purely technical metrics.

---

### A1 — Availability: Physical Supply Adequacy Under Gas Constraints

**GAS-1** quantifies the shadow price of gas deliverability to 
Nigeria's power sector across all four structural gas regime scenarios 
(baseline, upside, downside, shock_recovery) and tests whether that 
scarcity rent systematically exceeds domestic regulated benchmarks and 
LNG export parity. The model finds that gas shadow prices exceed LNG 
export benchmarks by 40–50x in all no-policy scenarios, confirming 
that Nigeria's gas constraint is physically driven, not price-driven — 
and that the power sector values gas at multiples of what any 
alternative use would pay.

**GAS-3** isolates whether NDC compliance cost and feasibility are 
determined by the level or the structural shape of gas decline — 
exponential depletion vs shock-recovery vs lag-ramp-uplift — using 
level-equivalent flat controls to disentangle the two effects. This 
distinguishes whether the NDC cost premium is a permanent structural 
feature of Nigeria's gas resource base or a transitional artefact of 
decline timing.

**STR-1** quantifies the role of battery storage in the optimal 
system and tests how the deployable hours parameter and solar surplus 
fraction determine whether storage complements or substitutes for EaaS 
solar deployment. Storage is modelled as a physical supply adequacy 
tool — an annual energy-shifting buffer that extends effective 
availability of intermittent solar supply — rather than as a 
standalone investment asset.

---

### A2 — Accessibility: Reliability of Supply Reaching Users

**REL-1** tests whether reliability feasibility under a fixed capital 
envelope depends primarily on the gas deliverability regime or the 
level of solar investment, and whether the answer changes across 
aggressive vs conservative solar build assumptions.

**REL-2** computes the marginal cost of tightening the reliability 
standard across gas regimes, testing whether the VoLL penalty in 
the objective or the explicit reliability constraint is the active 
driver of system reliability outcomes.

**REL-3** tests whether EaaS financing shifts the reliability 
feasibility frontier relative to public-capital-only financing under 
the same gas constraint, and whether this advantage persists when 
NDC 3.0 emissions targets are simultaneously imposed. This directly 
tests the claim that EaaS provides functional energy security benefit 
beyond financial convenience.

**DEM-1** tests whether the financing bottleneck diagnosis and NDC 
compliance cost findings are sensitive to the choice of served demand 
(23.08 TWh/yr, 2024 actual generation) versus latent demand 
(38–77 TWh/yr, reconstructed from electrification access rates) as 
the planning baseline — addressing whether the model's accessibility 
conclusions hold across the full range of plausible Nigerian demand 
conditions.

---

### A3 — Affordability: System Cost, Financing Structure, and Capital Adequacy

**FIN-1** (Phase 8 cost-emissions frontier analysis) characterises 
the full cost-emissions trade-off space across five scenario 
combinations — baseline, NDC3 public-only, and NDC3 EaaS under both 
unconditional and conditional ambition levels — computing marginal 
abatement costs and quantifying the system cost reduction from 
switching to EaaS financing at each NDC ambition level.

**FIN-2** derives and empirically verifies the minimum EaaS service 
tariff T* at which private capital becomes self-financing without 
public subsidy. T* is computed analytically as a function of solar 
CAPEX, required margin, and horizon-discounted energy yield, then 
confirmed across a tariff sweep grid. The conditional NDC scenario 
(required_margin=1.05, concessional blended finance) produces a 
strictly lower T* than the unconditional scenario (required_margin=1.10), 
directly quantifying the bankability benefit of concessional 
international climate finance.

**FIN-3** sweeps the public capital budget ceiling from 50% to 120% 
of B* to identify the budget level at which the capital bottleneck 
disappears without EaaS. A DisCo collection rate sensitivity 
(current=0.35 per NERC 2021–2023 average, reform_target=0.60 per 
World Bank DPO target, no_friction=1.00) grounds the budget analysis 
in Nigeria's institutional revenue reality — distinguishing whether 
EaaS is a planning convenience or a structural necessity under 
realistic public financing conditions.

**DEM-2** examines how demand growth rate uncertainty (low 2.5%, 
baseline 4%, high 6%) interacts with gas deliverability constraints 
to determine system cost risk, and whether higher demand growth 
amplifies or attenuates the marginal value of the EaaS financing 
mechanism.

---

### A4 — Acceptability: Long-Term Viability Under Emissions Targets

**GAS-2** tests whether EaaS solar deployment materially reduces the 
gas scarcity rent — establishing whether private capital deployment 
provides functional energy security relief by substituting for 
constrained gas, not merely repackaging the same investment under a 
different ownership structure. A positive result means EaaS is an 
energy security instrument as well as a financing instrument; a null 
result means its value is purely financial.

**POL-1** compares the investment trajectory and system cost under 
NDC 3.0 (proportional top-down apportionment) unconditional and 
conditional targets, with NDC 2.0 BAU-relative targets as a 
methodological robustness arm. Tests whether the EaaS mechanism 
absorbs or amplifies methodology-driven cost variation.
---

### Dual-Based Scarcity Valuation — Cross-Cutting Contribution

Across all four dimensions, the model extracts LP dual variables 
(shadow prices) for the gas availability constraint, the public 
capital budget constraint, the annual emissions cap, and the 
reliability constraint. These shadow prices provide a unified 
economic language for interpreting system bottlenecks: when the gas 
shadow price exceeds LNG export parity (GAS-1), when the budget 
shadow is positive (FIN-3), when the carbon shadow rises steeply 
(POL-1), and when the reliability dual diverges across gas regimes 
(REL-2), they collectively identify which constraint is actually 
binding the system at each point in the planning horizon — and what 
relaxing each constraint would be worth.

## 🏗 Repository Structure
```
thesis-repo/
│
├── data/                                 # Clean and processed datasets
│   ├── demand/                           # Demand projections, load profiles
│   ├── gas/                              # Reservoir decline inputs, analogs
│   ├── solar/                            # PV irradiance datasets
│   ├── cost/                             # CAPEX/OPEX tables
│   ├── stochastic/                       # Probability distributions & samples
│   └── README.md                         # Data provenance and licenses
|
|
├── src/                                  # Core modeling modules
│   ├── gas_supply.py                     # Gas availability loading
│   ├── demand.py                         # Baseline demand projection
│   ├── solar.py                          # PV generation + capacity factors
│   ├── storage.py                        # Stateful SOC, dispatch simulation
│   ├── dispatch.py                       # System dispatch & constraints
│   ├── optimize_model.py                 # model construction + solve only
│   ├── optimize_experiments.py           # experiment orchestration (Pareto, sweeps, diagnostics)
│   ├── stochastic.py                     # Monte Carlo wrapper
│   ├── scenarios.py                      # Scenario configuration
│   ├── economics.py                      # Economic Evaluation Utilities 
│   ├── EaaS.py                           # transforms system parameters
│   ├── gas_balance.py
│   ├── io.py                             # 
│   ├── postprocess.py                    # shadow price benchmarking
│   ├── build_gas_deliverability.py       # Gas decline & deliverability construction
│   └── utils.py                          # Shared utilities / helpers
|   
│
├── scripts/
│   ├── 00_build_emissions_cap.py                   # Build NDC 3.0 (proportional) + NDC 2.0 (BAU-relative) caps
│   ├── 01_run_baseline.py                          # Unconstrained baseline (no NDC, no EaaS)
│   ├── 02_run_eaas.py                              # EaaS baseline (no NDC)
│   ├── 02_run_ndc_caps.py                          # NDC3 public-only financing
│   ├── 03_run_ndc_eaas.py                          # NDC3 with EaaS financing
│   ├── 04_run_fin2_tariff_sweep.py                 # FIN-2: EaaS tariff bankability threshold
│   ├── 05_run_fin3_capital_sweep.py                # FIN-3: Capital budget × collection rate
│   ├── 06_run_gas1_shadow_benchmarks.py            # GAS-1: Gas scarcity rent vs benchmarks
│   ├── 07_run_gas2_eaas_gas_relief.py              # GAS-2: EaaS gas constraint relief
│   ├── 08_phase8_analysis.py                       # Phase 8: Cost-emissions frontier
│   ├── 08_run_gas3_regime_ndc_feasibility.py       # GAS-3: Regime shape vs NDC
│   ├── 09_run_rel1_feasibility.py                  # REL-1: Reliability feasibility matrix
│   ├── 10_run_rel2_marginal_cost.py                # REL-2: Marginal cost of reliability
│   ├── 11_run_rel3_financing_frontier.py           # REL-3: EaaS reliability frontier
│   ├── 12_run_dem1_demand_sensitivity.py           # DEM-1: Demand sensitivity
│   ├── 13_run_dem2_growth_gas_interaction.py       # DEM-2: Growth-gas interaction
│   ├── 14_run_str1_storage_role.py                 # STR-1: Storage role and regime
│   ├── 15_run_pol1_ndc_comparison.py               # POL-1: NDC 2.0 vs NDC 3.0
│   ├── build_gas_deliverability.py                 # Construct gas deliverability scenarios
│   ├── plot_storage_regime.py                      # Visualise storage constraint binding
│   ├── 16_plot_cost_decomposition.py               # System cost decomposition figure
│   ├── 99_validate_baseline.py                     # LP output vs 2024 observed validation
│   ├── 09b_run_rel1_mode_sensitivity.py            # REL-1 robustness: annual vs total mode
│   ├── 09b_run_rel1_retirement_sensitivity.py      # REL-1 robustness: gas fleet retirement rate|
│   └── plot_system_diagnostics.py                  # Visualise system evolution + shadow prices|
│
│
├── notebooks/                                      # Planned - not yet populated
│   ├── 01_data_validation.ipynb
│   ├── 02_model_sanity_checks.ipynb
│   ├── 03_baseline_results.ipynb
│   ├── 04_pareto_analysis.ipynb
│   ├── 05_shadow_price_analysis.ipynb
│   ├── 06_reliability_analysis.ipynb
│   ├── 07_eaas_analysis.ipynb
│   ├── 08_stochastic_results.ipynb
│   ├── 09_policy_synthesis.ipynb
│   └── 99_appendix_figures.ipynb│
|
|
├── results/
│   ├── baseline/              # 01_run_baseline.py outputs
│   ├── eaas/                  # 02_run_eaas.py outputs
│   ├── ndc/                   # 02_run_ndc_caps.py outputs
│   ├── ndc_eaas/              # 03_run_ndc_eaas.py outputs
│   ├── phase8/                # 08_phase8_analysis.py outputs
│   ├── fin2/                  # FIN-2 tariff sweep outputs
│   ├── fin3/                  # FIN-3 capital sweep outputs
│   ├── gas1/                  # GAS-1 shadow benchmark outputs
│   ├── gas2/                  # GAS-2 EaaS gas relief outputs
│   └── [gas3/ rel1/ ... pol1/ populated by remaining RQ scripts]
|
|
├── docs/                    # Draft thesis chapters, reports, policy briefs
│   ├── thesis/
│   ├── policy_brief/
│   ├── investor_note/
│   └── presentation/
|
|
├── tests/                   # Integration tests (model smoke, cap binding, EaaS activation)
│
├── environment.yml          # Full environment specification
├── requirements.txt         # Python dependencies
├── LICENSE                  # License for code and reproducibility
└── README.md                # This file
```

## ⚙️ Software & Dependencies
This project uses the following major packages:
* Python 3.10+
* Pyomo 
* NumPy, Pandas
* SciPy
* Matplotlib / Plotly
Full environment details:
* requirements.txt: pip installation
* environment.yml: Conda environment with solver support

## ▶️ How to Reproduce the Model

Step 1: Clone the repository
git clone https://github.com/I-Sobe/thesis-code.git
cd thesis-code

Step 2: Create environment
conda env create -f environment.yml
conda activate nigeria-energy

Step 3: Build gas deliverability scenarios
python scripts/build_gas_deliverability.py

Step 4: Run baseline (must run first — emissions caps depend on it)
python scripts/01_run_baseline.py

Step 5: Build emissions caps (reads baseline diagnostics)
python scripts/00_build_emissions_cap.py

Step 6: Validate baseline against observed 2024 data
python scripts/99_validate_baseline.py

Step 7: Run core NDC and EaaS scenarios
python scripts/02_run_eaas.py
python scripts/02_run_ndc_caps.py
python scripts/03_run_ndc_eaas.py

Step 8: Run financing research questions
python scripts/04_run_fin2_tariff_sweep.py
python scripts/05_run_fin3_capital_sweep.py

Step 9: Run gas regime research questions
python scripts/06_run_gas1_shadow_benchmarks.py
python scripts/07_run_gas2_eaas_gas_relief.py
python scripts/08_run_gas3_regime_ndc_feasibility.py

Step 10: Run reliability research questions
python scripts/09_run_rel1_feasibility.py
python scripts/09b_run_rel1_mode_sensitivity.py
python scripts/09b_run_rel1_retirement_sensitivity.py
python scripts/10_run_rel2_marginal_cost.py
python scripts/11_run_rel3_financing_frontier.py

Step 11: Run demand and storage research questions
python scripts/12_run_dem1_demand_sensitivity.py
python scripts/13_run_dem2_growth_gas_interaction.py
python scripts/14_run_str1_storage_role.py

Step 12: Run policy comparison and aggregation
python scripts/15_run_pol1_ndc_comparison.py
python scripts/08_phase8_analysis.py

Step 13: Generate cost decomposition figure
python scripts/16_plot_cost_decomposition.py

Results are written to results/<script_name>/ directories as CSV and JSON.

## 📊 Outputs
The model produces:
* Optimal generation mixes
* Capacity expansion schedules
* Cost and LCOE trajectories
* Emissions pathways
* Reliability metrics (Unserved energy (TWh) and derived reliability metrics (annual and horizon-level adequacy))
* Pareto frontiers for cost–emissions–reliability
* Monte Carlo distributions (LCOE, reliability, solar share)
* EaaS bankability threshold (T*) and financing gap quantification
* DisCo collection rate sensitivity on public capital adequacy
* Gas scarcity rent vs domestic and export opportunity cost benchmarks
Accessible in (to be populated):
```
results/deterministic/
results/stochastic/
results/figures/
```
## 📄 Reproducibility
This repository follows FAIR principles and includes:
* Full environment specifications
* Data provenance documentation
* Unit tests for core modules
* Clear module-level docstrings
* Notebook-based reproducibility pipeline
Academic examiners, supervisors, and collaborators can fully replicate all results using the included instructions.

## 🧩 Associated Documents
* Full thesis draft (in docs/thesis/)
* Policy brief for Nigerian regulators - planned
* Investor notes for EaaS providers and DFIs - planned
* Conference paper - planned
* Defense presentation slides - planned

## 👤 Author
Nwangene Sobe-Olisa Andrew
MSc Petroleum & Energy Resource Engineering
Specialization: Energy Systems Strategy, Techno-Economic Modeling, Power System Optimization
