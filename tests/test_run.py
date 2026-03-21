import pandas as pd
from src.scenarios import load_scenario
from src.optimize_experiments import generate_epsilon_pareto

def to_float(x):
    return float(str(x).replace(",", "").replace("$", "").strip())

scenario = load_scenario(
    demand_level_case="served",
    demand_case="baseline",
    gas_deliverability_case="baseline",
    carbon_case="no_policy",
    start_year=2025,
    end_year=2045,
)

econ = {}

gas_cost_df = pd.read_csv("data/cost/processed/gas_cost.csv", thousands=",")
gas_low_row = gas_cost_df[gas_cost_df["Scenario"] == "gas_low"].iloc[0]
econ["GAS_COST_PER_TWH_TH"] = to_float(gas_low_row["total_usd_per_twh_th"])

solar_df = pd.read_csv("data/cost/processed/solar_capex.csv", thousands=",")
solar_row = solar_df[(solar_df["Scenario"] == "solar_low") & (solar_df["Year"] == 2025)].iloc[0]
econ["SOLAR_CAPEX_PER_MW"] = to_float(solar_row["Solar_capex_usd_per_mw"])

storage_df = pd.read_csv("data/cost/processed/storage_capex.csv", thousands=",")
storage_row = storage_df[(storage_df["Scenario"] == "Storage_low") & (storage_df["Year"] == 2025)].iloc[0]
econ["STORAGE_COST_PER_MWH"] = to_float(storage_row["Storage_capex_usd_per_mwh"])

voll_df = pd.read_csv("data/cost/processed/unserved_energy_penalty.csv", thousands=",")
voll_row = voll_df[(voll_df["scenario"] == "voll_low") & (voll_df["year"] == 2025)].iloc[0]
econ["UNSERVED_ENERGY_PENALTY"] = to_float(voll_row["voll_usd_per_twh"])

econ["CARBON_EMISSION_FACTOR"] = 0.421

results = generate_epsilon_pareto(scenario=scenario, econ=econ, emissions_caps=[1e18])
out0 = results["results"][0]
dv = out0["decision_variables"]
diag = out0["diagnostics"]

print("Solar addition (MW/year):", dv["solar_add_mw_by_year"])
print("Storage capacity (MWh):", dv["final_storage_capacity_mwh"])
print("Storage capacity (MWh):", dv["storage_capacity_mwh"])
print("Unserved 2025 (TWh):", diag["unserved_twh_by_year"][2025])
print("Gas to power 2025 (TWh_th):", diag["gas_to_power_twh_th_by_year"][2025])
