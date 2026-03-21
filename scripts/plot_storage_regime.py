import json
import matplotlib.pyplot as plt
from pathlib import Path

# path to diagnostics output
RESULT_PATH = Path("results/ndc/ndc_unconditional_20_voll_low/diagnostics.json")

with open(RESULT_PATH) as f:
    diag = json.load(f)

binding = diag["storage_binding_by_year"]

years = [int(y) for y in binding.keys()]
values = list(binding.values())

mapping = {
    "none":0,
    "power_limit":1,
    "cycle_limit":2,
    "solar_limit":3
}

y = [mapping[v] for v in values]

plt.figure(figsize=(8,4))

plt.step(years, y, where="mid")

plt.yticks(
    [0,1,2,3],
    ["none","power limit","cycle limit","solar limit"]
)

plt.xlabel("Year")
plt.ylabel("Binding Storage Constraint")
plt.title("Storage Constraint Regime Over Time")

plt.grid(True)

plt.tight_layout()

plt.show()