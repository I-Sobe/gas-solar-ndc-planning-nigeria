import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("results/plots", exist_ok=True)


# ----------------------------
# Load data
# ----------------------------
CASES = {
    "baseline":             "results/baseline/timeseries.csv",
    "ndc3_unconditional":   "results/ndc/ndc3_unconditional_voll_mid/timeseries.csv",
    "ndc3_conditional":     "results/ndc/ndc3_conditional_voll_mid/timeseries.csv",
    "ndc3_unconditional_eaas": "results/ndc_eaas/ndc3_unconditional_eaas_voll_mid/timeseries.csv",
"ndc3_conditional_eaas":   "results/ndc_eaas/ndc3_conditional_eaas_voll_mid/timeseries.csv",
}

# Load all cases into a dict of DataFrames
dfs = {name: pd.read_csv(path) for name, path in CASES.items()}

# Use baseline as the primary single-case reference for panels A and B
df = dfs["baseline"]

years = df["year"]

# Panel A (system behavior)
solar = df["solar_generation_twh_e"]
gas = df["gas_generation_twh_e"]
storage = df["storage_discharge_twh_e"]

# Panel B (shadow prices)
if "gas_shadow_usd_per_twh_th" in df.columns:
    gas_shadow = df["gas_shadow_usd_per_twh_th"]
else:
    gas_shadow = df["gas-shadow_price_usd_per_twh_th"]

carbon_shadow = df["carbon_shadow_usd_per_tco2"]

# Optional VoLL (set or load)
VOLL = 10_000_000_000 # voll_mid: 10B USD/TWH

# ----------------------------
# Create figure
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ----------------------------
# Panel A: System response
# ----------------------------
ax1.plot(years, solar, label="Solar generation")
ax1.plot(years, gas, label="Gas generation")
ax1.plot(years, storage, label="Storage discharge")

# Overlay all cases on Panel A for comparison
COLOURS = {
    "baseline": "grey",
    "ndc3_unconditional": "orange",
    "ndc3_conditional": "green",
    "ndc3_unconditional_eaas": "red",
    "ndc3_conditional_eaas": "purple",
}
for case_name, case_df in dfs.items():
    ax1.plot(
        case_df["year"],
        case_df["solar_generation_twh_e"],
        color=COLOURS[case_name],
        label=f"Solar — {case_name}",
        linestyle="--" if "eaas" in case_name else "-",
    )

ax1.set_xlabel("Year")
ax1.set_ylabel("TWh")
ax1.set_title("System Evolution")

ax1.legend()
ax1.grid()

# ----------------------------
# Panel B: Marginal prices
# ----------------------------

for case_name, case_df in dfs.items():
    ax2.plot(
        case_df["year"],
        pd.to_numeric(case_df["carbon_shadow_usd_per_tco2"], errors="coerce"),
        color=COLOURS[case_name],
        label=f"Carbon shadow — {case_name}",
        linestyle="--" if "eaas" in case_name else "-",
    )

ax2.axhline(VOLL, linestyle="--", label="VoLL_USD_PER_TWH")

ax2.set_xlabel("Year")
ax2.set_ylabel("USD")
ax2.set_title("Marginal Value of Constraints")

# Compute regime transition years from data
gas_binding_year = df.loc[
    (df["gas_shadow_usd_per_twh_th"] if "gas_shadow_usd_per_twh_th" in df.columns
     else df["gas_shadow_price_usd_per_twh_th"]) > 1e-6,
    "year"
].min()

carbon_binding_year = df.loc[
    df["carbon_shadow_usd_per_tco2"] > 1e-6,
    "year"
].min()

if pd.notna(gas_binding_year):
    ax2.axvline(gas_binding_year, linestyle=":", label=f"Gas shadow binds ({int(gas_binding_year)})")
if pd.notna(carbon_binding_year):
    ax2.axvline(carbon_binding_year, linestyle=":", label=f"Carbon shadow binds ({int(carbon_binding_year)})")

ax2.legend()
ax2.grid()

# ----------------------------
# Save + show
# ----------------------------
plt.tight_layout()
plt.savefig("results/plots/system_diagnostics.png", dpi=300)
plt.show()