import sys
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

import json
from src.scenarios import load_scenario
from src.stochastic import run_stochastic_simulation, compute_risk_metrics
from src.optimize_model import build_model, solve_model
from src.io import load_econ

from pyomo.environ import value

# path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# -------------------------------------------------
# Load inputs
# -------------------------------------------------

scenario = load_scenario()
econ = load_econ()

# -------------------------------------------------
# Solve deterministic optimal system
# -------------------------------------------------

m = build_model(scenario, econ)
solve_model(m)

# Extract optimal capacity paths
capacity_paths = {
    "solar_mw": [value(m.solar_capacity_mw[t]) for t in range(len(scenario["years"]))],
    "storage_mwh": [value(m.storage_capacity_mwh[t]) for t in range(len(scenario["years"]))],
}

# -------------------------------------------------
# Monte Carlo uncertainty analysis
# -------------------------------------------------

outcomes = run_stochastic_simulation(
    base_scenario=scenario,
    econ=econ,
    capacity_paths=capacity_paths,
    carbon_mu=3.9,
    carbon_sigma=0.4,
    N=5000,
    seed=42
)

costs = [c for _, c in outcomes]
risk = compute_risk_metrics(costs)
#risk = compute_risk_metrics(outcomes)

# saving result
with open("results/uncertainty_metrics.json","w") as f:
    json.dump(risk, f, indent=2)
# -------------------------------------------------
# Print results
# -------------------------------------------------
print("\nExpected cost:", risk["expected"])
print("Variance:", risk["variance"])
print("VaR (95%):", risk["VaR"])
print("CVaR (95%):", risk["CVaR"])
print("\nMonte Carlo Risk Metrics\n")
print(risk)

groups = defaultdict(list)
for regime, cost in outcomes:
    groups[regime].append(cost)

for regime, vals in groups.items():
    plt.hist(vals, bins=30, alpha=0.5, label=regime)

plt.axvline(risk["VaR"], color="red", linestyle="--", label="VaR")
plt.axvline(risk["CVaR"], color="black", linestyle="--", label="CVaR")

plt.xlabel("Total System Cost (USD)")
plt.ylabel("Frequency")
plt.title("Cost Distribution by Gas Supply Regime")
plt.legend()
plt.show()