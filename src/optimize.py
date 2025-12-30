"""
Optimization Module

Constructs and solves the multi-objective optimization problem
using ε-constraint or weighted-sum approaches. Produces
Pareto-optimal solutions for cost, emissions, and reliability.

Functions:
    build_model(inputs)
    solve_model(model, solver='cbc')
    generate_pareto_front(...)
"""

"""
optimize.py
Deterministic system evaluation and cost calculation
"""

import numpy as np

from src.demand import project_baseline_demand
from src.gas_supply import gas_generation_cap
from src.solar import solar_generation, solar_capacity_trajectory
from src.storage import BatteryStorage
from src.dispatch import dispatch_energy


# ----------------------------
# COST PARAMETERS (BASELINE)
# ----------------------------
GAS_COST_PER_TWH = 50e6        # $/TWh
SOLAR_CAPEX_PER_MW = 800_000   # $/MW
CARBON_EMISSION_FACTOR = 0.4  # tCO2/MWh
UNSERVED_ENERGY_PENALTY = 1e9 # $/TWh (Value of Lost Load)


