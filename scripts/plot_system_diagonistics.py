import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("results/annualcap_ndc_unconditional_20_timeseries.csv")

years = df["year"]

# Panel A (system behavior)
solar = df["solar_generation_twh_e"]
gas = df["gas_generation_twh_e"]
storage = df["storage_discharge_twh_e"]

# Panel B (shadow prices)
gas_shadow = df["gas_shadow_usd_per_twh_th"]
carbon_shadow = df["carbon_shadow_usd_per_tco2"]

# Optional VoLL (set or load)
VOLL = 1000

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

ax1.set_xlabel("Year")
ax1.set_ylabel("TWh")
ax1.set_title("System Evolution")

ax1.legend()
ax1.grid()

# ----------------------------
# Panel B: Marginal prices
# ----------------------------
ax2.plot(years, gas_shadow, label="Gas scarcity price")
ax2.plot(years, carbon_shadow, label="Carbon shadow price")

ax2.axhline(VOLL, linestyle="--", label="VoLL")

ax2.set_xlabel("Year")
ax2.set_ylabel("USD")
ax2.set_title("Marginal Value of Constraints")

ax2.axvline(2032, linestyle=":", label="Gas scarcity onset")
ax2.axvline(2035, linestyle=":", label="Carbon constraint binds")

ax2.legend()
ax2.grid()

# ----------------------------
# Save + show
# ----------------------------
plt.tight_layout()
plt.savefig("results/system_diagnostics.png", dpi=300)
plt.show()