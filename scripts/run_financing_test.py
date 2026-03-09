import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.optimize_experiments import run_financing_vs_resource_test

def compute_reliability_metrics(diag):

    demand = diag["demand_twh_by_year"]
    unserved = diag["unserved_twh_by_year"]

    years = sorted(demand.keys())

    # ----- annual reliability -----
    annual = {}
    for y in years:
        d = demand[y]
        u = unserved[y]

        if d <= 0:
            annual[y] = None
        else:
            annual[y] = 1.0 - (u / d)

    # ----- worst year reliability -----
    worst = min(r for r in annual.values() if r is not None)

    # ----- horizon reliability -----
    total_demand = sum(demand.values())
    total_unserved = sum(unserved.values())

    horizon = 1.0 - (total_unserved / total_demand)

    return {
        "annual_reliability_by_year": annual,
        "worst_year_reliability": worst,
        "horizon_reliability": horizon,
        "expected_unserved_energy_twh": total_unserved
    }

# Economics parameters (match your model inputs)
econ = {
    "GAS_COST_PER_TWH_TH": 35000000,
    "SOLAR_CAPEX_PER_MW": 1456000,
    "STORAGE_COST_PER_MWH": 350000,
    "UNSERVED_ENERGY_PENALTY": 20000000000,
    "CARBON_EMISSION_FACTOR": 0.35
}

results = run_financing_vs_resource_test(econ)

for r in results:
    print(r)