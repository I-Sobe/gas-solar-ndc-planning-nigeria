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
if "carbon_shadow_usd_per_tco2" in df.columns:
    carbon_shadow = df["carbon_shadow_usd_per_tco2"]
else:
    carbon_shadow = df["caron_shadow_price_usd_per_tco2"]
# Optional VoLL (set or load)
VOLL = 10_000_000_000 # voll_mid: 10B USD/TWH

# ----------------------------
# Create figure
# ----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

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

# ----------------------------
# Panel B: Marginal prices
# ----------------------------

# ax2: carbon shadow only
for case_name, case_df in dfs.items():
    cs_col = "carbon_shadow_usd_per_tco2" if "carbon_shadow_usd_per_tco2" in case_df.columns else "carbon_shadow_price_usd_per_tco2"
    ax2.plot(
        case_df["year"],
        pd.to_numeric(case_df[cs_col], errors="coerce"),
        color=COLOURS[case_name],
        label=f"Carbon shadow — {case_name}",
        linestyle="--" if "eaas" in case_name else "-",
    )
ax2.set_xlabel("Year")
ax2.set_ylabel("USD / tCO2")
ax2.set_title("Carbon Shadow Price")
ax2.legend()
ax2.grid()

# ax3: gas shadow + VoLL (same unit: USD/TWh)
gs_col = "gas_shadow_price_usd_per_twh_th" if "gas_shadow_price_usd_per_twh_th" in df.columns else "gas_shadow_usd_per_twh_th"
ax3.plot(years, df[gs_col], label="Gas shadow — baseline", color="grey")
ax3.axhline(VOLL, linestyle="--", color="blue", label="VoLL (10B USD/TWh)")
if pd.notna(gas_binding_year):
    ax3.axvline(gas_binding_year, linestyle=":", label=f"Gas binds ({int(gas_binding_year)})")
ax3.set_xlabel("Year")
ax3.set_ylabel("USD / TWh_th")
ax3.set_title("Gas Scarcity Rent vs VoLL")
ax3.legend()
ax3.grid()

ax2.set_xlabel("Year")
ax2.set_ylabel("USD")
ax2.set_title("Marginal Value of Constraints")


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