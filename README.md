# README.md
National-Scale Techno-Economic Optimization of Hybrid Gas–Solar Systems Under Reservoir Decline and Energy-as-a-Service (EaaS) Deployment Scenarios — A Stochastic Multi-Objective Modeling Framework

## 🔍 Project Overview
This repository contains the full modeling framework, datasets, scripts, and reproducibility instructions for the MSc thesis:

“Techno-Economic Optimization of Gas–Solar Power Systems Under Reservoir Decline and Energy-as-a-Service Deployment: A Stochastic Multi-Objective Framework for Nigeria’s Energy Transition.”

The project combines reservoir-constrained gas supply modeling, utility-scale solar and storage simulation, energy system optimization, and EaaS demand-side interventions into a unified national planning tool.
It is designed to evaluate long-term energy security risks associated with declining gas fields while exploring the role of distributed and utility-scale solar in supporting Nigeria’s reliability, cost, and carbon reduction objectives.
This work supports insights for:
* National planners and regulators (MoP/NERC)
* Utilities and generation companies
* Renewable IPPs
* EaaS providers and commercial/industrial customers
* International development partners (WB, AfDB, IRENA)

## 🧠 Core Contributions
This project achieves four technical and strategic contributions:
### Reservoir-Integrated Supply Modeling
A representative gas field supplying the power sector is modeled using Arps decline formulations, constraining annual system-level gas availability.
### Stochastic Multi-Objective Energy System Optimization
A Pyomo/PyPSA-based optimization framework evaluates trade-offs between cost, emissions, and reliability using ε-constraint Pareto generation.
### Monte Carlo Risk Analysis
Gas price, carbon price, and solar CAPEX uncertainties are incorporated through sampling (N=200–500), producing risk-adjusted planning metrics (CVaR, probability of shortage).
### Demand-Side EaaS Integration
Energy-as-a-Service business models are incorporated as demand-modifying and reserve-support modules, enabling assessment of distributed energy’s contribution to national supply adequacy.

## 🏗 Repository Structure
```
thesis-repo/
│
├── data/                    # Clean and processed datasets
│   ├── demand/              # Demand projections, load profiles
│   ├── gas/                 # Reservoir decline inputs, analogs
│   ├── solar/               # PV irradiance datasets
│   ├── cost/                # CAPEX/OPEX tables
│   ├── stochastic/          # Probability distributions & samples
│   └── README.md            # Data provenance and licenses
│
├── src/                     # Core modeling modules
│   ├── gas_supply.py        # Decline curve model (Arps)
│   ├── demand.py            # Baseline + EaaS-adjusted demand
│   ├── solar.py             # PV generation + capacity factors
│   ├── storage.py           # Battery SOC formulation
│   ├── dispatch.py          # System dispatch & constraints
│   ├── optimize.py          # Multi-objective optimization
│   ├── stochastic.py        # Monte Carlo wrapper
│   ├── scenarios.py         # Scenario configuration
│   └── utils.py             # Shared utilities / helpers
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