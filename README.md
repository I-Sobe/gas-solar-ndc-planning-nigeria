# README.md
National-Scale Techno-Economic Optimization of Hybrid Gas–Solar Systems Under Delivered Gas-to-Power Constraints and Energy-as-a-Service (EaaS) Deployment Scenarios — A Stochastic Multi-Objective Modeling Framework.

## 🔍 Project Overview
This repository contains the full modeling framework, datasets, scripts, and reproducibility instructions for the MSc thesis:

“Techno-Economic Optimization of Gas–Solar Power Systems Under Gas-to-power deliverability constraints and Energy-as-a-Service Deployment: A Stochastic Multi-Objective Framework for Nigeria’s Energy Transition.”

The project combines delivered gas-to-power availability modeling (informed by upstream decline, infrastructure, and policy constraints), utility-scale solar and storage simulation, energy system optimization, and Energy-as-a-Service (EaaS) deployment and financing mechanisms into a unified national planning tool.
It is designed to evaluate long-term energy security risks associated with declining gas fields while exploring the role of distributed and utility-scale solar in supporting Nigeria’s reliability, cost, and carbon reduction objectives.
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
  - gas-fired generation (subject to availability constraints),
  - utility-scale solar photovoltaic generation,
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
- Decline analytics (e.g., Arps-style profiles) are used to inform **aggregate deliverability trends**.
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
Electricity reliability is represented through **unserved energy**, calculated annually and penalized economically in the objective function.

This allows reliability to be:
- explicitly traded against cost and emissions,
- evaluated across gas-availability, policy, and financing scenarios.


### Architectural Mapping: Text → Code
Key modeling concepts map directly to repository modules:

| Concept | Module |
|------|------|
| Gas-to-power availability | `src/gas_supply.py` |
| Policy and availability scenarios | `src/scenarios.py` |
| Demand projection | `src/demand.py` |
| Solar generation | `src/solar.py` |
| Battery storage | `src/storage.py` |
| Reliability and dispatch | `src/dispatch.py` |
| Cost, emissions, financing | `src/economics.py` |
| Optimization formulation | `src/optimize_model.py` |
| Experiment orchestration | `src/optimize_experiments.py` |
| Visualization and Pareto analysis | `notebooks/visualization.ipynb` |

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
This project delivers four technical and strategic contributions to energy system planning for gas-dominated emerging economies:

### Delivered Gas-to-Power Availability Modeling Informed by Decline and Policy
The study introduces an aggregate delivered gas-to-power availability constraint, informed by decline analytics, midstream infrastructure performance, and domestic gas allocation regimes. Decline formulations (e.g., Arps-style profiles) are used to parameterize deliverability envelopes, without attempting reservoir or field-level simulation.

### Multi-Objective Energy System Optimization under Reliability Constraints
A Python-based optimization framework (implemented using Pyomo) evaluates trade-offs between total system cost, carbon emissions, and electricity reliability. Pareto-efficient solution sets are generated using weighted-sum and ε-constraint methods, enabling transparent assessment of planning trade-offs.

### Stochastic Risk and Uncertainty Analysis for Long-Term Planning
Key uncertainties—gas prices, carbon prices, and technology costs—are incorporated through Monte Carlo sampling and structured scenario analysis. The framework produces risk-aware planning insights, including probability-weighted outcomes for cost, emissions, and supply adequacy.

### Deployment and Financing Pathway Analysis via Energy-as-a-Service
Energy-as-a-Service (EaaS) models are incorporated strictly as deployment and financing mechanisms, affecting cost of capital, feasible deployment rates, and investment constraints for solar and storage. This enables assessment of how alternative ownership and financing structures influence scalability and system adequacy, without modeling customer behavior, demand-side efficiency, or contract-level dynamics.

## 🏗 Repository Structure
```
thesis-repo/
│
├── data/                           # Clean and processed datasets
│   ├── demand/                     # Demand projections, load profiles
│   ├── gas/                        # Reservoir decline inputs, analogs
│   ├── solar/                      # PV irradiance datasets
│   ├── cost/                       # CAPEX/OPEX tables
│   ├── stochastic/                 # Probability distributions & samples
│   └── README.md                   # Data provenance and licenses
│
├── src/                            # Core modeling modules
│   ├── gas_supply.py               # Decline curve model (Arps)
│   ├── demand.py                   # Baseline + EaaS-adjusted demand
│   ├── solar.py                    # PV generation + capacity factors
│   ├── storage.py                  # Battery SOC formulation
│   ├── dispatch.py                 # System dispatch & constraints
│   ├── optimize_model.py           # model construction + solve only
│   ├── optimize_experiments.py     # model construction + solve only
│   ├── stochastic.py               # Monte Carlo wrapper
│   ├── scenarios.py                # Scenario configuration
│   ├── economics.py                # Economic Evaluation Utilities 
│   ├── EaaS.py                     # transforms system parameters
│   └── utils.py                    # Shared utilities / helpers
|   
│
├── notebooks/               # Reproducible analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_prototype_model.ipynb
│   ├── 03_optimization_runs.ipynb
│   ├── 04_stochastic_analysis.ipynb
│   ├── 05_visualizations.ipynb
│   └── 99_appendix_figures.ipynb
│
├── results/                 # Generated figures, tables, scenario outputs
│   ├── deterministic/
│   ├── stochastic/
│   └── figures/
│
├── docs/                    # Draft thesis chapters, reports, policy briefs
│   ├── thesis/
│   ├── policy_brief/
│   ├── investor_note/
│   └── presentation/
│
├── tests/                   # Unit and integration tests
│
├── environment.yml          # Full environment specification
├── requirements.txt         # Python dependencies
├── LICENSE                  # License for code and reproducibility
└── README.md                # This file
```

## ⚙️ Software & Dependencies
This project uses the following major packages:
* Python 3.10+
* Pyomo / PyPSA
* NumPy, Pandas
* SciPy
* Matplotlib / Plotly
* Salib (global sensitivity, if used)
* tqdm, joblib (parallel Monte Carlo)
Full environment details:
* requirements.txt: pip installation
* environment.yml: Conda environment with solver support

## ▶️ How to Reproduce the Model
Step 1: Clone the repository
git clone https://github.com/I-Sobe/thesis-code.git
cd thesis-code

Step 2: Create environment
Using Conda
conda env create -f environment.yml
conda activate nigeria-energy

Step 3: Run prototype model
python src/optimize.py --scenario baseline

Step 4: Run deterministic scenario matrix
python src/scenarios.py --run all_deterministic

Step 5: Ru stochastic wrapper
python src/stochastic.py --scenario high_decline --samples 300

Step 6: Reporduce all figures
Open and execute:
notebooks/05_visualizations.ipynb

## 📊 Outputs
The model produces:
* Optimal generation mixes
* Capacity expansion schedules
* Cost and LCOE trajectories
* Emissions pathways
* Reliability metrics (LOLP / unmet load)
* Pareto frontiers for cost–emissions–reliability
* Monte Carlo distributions (LCOE, reliability, solar share)
* EaaS impact quantification (demand reduction, reserve contribution)
Accessible in:
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
* Policy brief for regulators
* Investor memo for EaaS providers
* Conference paper draft
* Defense presentation slides

## 👤 Author
Nwangene Sobe-Olisa Andrew
MSc Petroleum & Energy Resource Engineering
Specialization: Energy Systems Strategy, Techno-Economic Modeling, Power System Optimization
