# Step 1 — Grid generation per capita, Nigeria 2010–2025

**Status:** complete (revised — connected-population basis added).
Feeds correction plan Phase 1.6, Step 1.
**Workbook:** `step1_per_capita_diagnostic.xlsx`

---

## Question

Does observed growth in Nigerian grid generation carry information about growth in
electricity **demand**, or only about **population growth**, **access expansion**,
and the persistence of supply constraints?

This matters because Phase 1.6 must source a demand-growth rate `g`, and the obvious
candidate — historical generation CAGR — is only admissible if the series reflects
demand rather than delivery.

---

## Result 1 — total-population basis

`(1 + g_gen) = (1 + g_pop) × (1 + g_percap)`

| Window | Generation | Population | Per capita | Demographic share |
|---|---|---|---|---|
| 2010–2025 | 3.14% | 2.39% | **+0.73%** | 76% |
| 2013–2025 | 3.06% | 2.29% | **+0.76%** | 75% |
| 2015–2025 | 2.28% | 2.22% | **+0.05%** | 98% |
| 2017–2025 | 3.22% | 2.16% | **+1.04%** | 67% |
| 2014–2024 | 1.53% | 2.27% | **−0.73%** | >100% |

Per-capita grid generation moved from ~157 to ~175 kWh/person over fifteen years —
about **2.1 W of additional continuous supply per person**, and negative under two
of five windows. Population growth is stable at 2.2–2.4% on every window; the
per-capita residual is not, ranging from −0.73% to +1.04%.

## Result 2 — connected-population basis (the stronger finding)

Tier 1's correct denominator is *connected* population, not total. Using total
population folds access expansion into what is being called organic growth.

`(1 + g_gen) = (1 + g_connected_pop) × (1 + g_per_connected)`

| Window | Generation | Access rate | Connected pop | **kWh per connected person** |
|---|---|---|---|---|
| 2010–2024 | 2.63% | +1.90% | +4.36% | **−1.66%** |
| 2013–2024 | 2.40% | +1.07% | +3.40% | **−0.97%** |
| 2015–2024 | 1.39% | +1.96% | +4.24% | **−2.73%** |
| 2017–2024 | 2.21% | +2.00% | +4.21% | **−1.93%** |
| 2010–2023 | 3.51% | +1.89% | +4.37% | **−0.82%** |

Grid generation per connected person **fell from ~327 to ~258 kWh/year between 2010
and 2024, a decline of 21%**. It is negative on every window tested.

Connected population grew at 3.4–4.4%/yr — *faster* than generation on every window.
Nigeria connected people to the grid more quickly than it added electricity to send
them.

---

## Interpretation

**On total population, 67–100%+ of observed generation growth is demographic. On
connected population, all of it is demographic and then some: intensity per
connected person declined throughout.**

The series therefore carries no usable signal about consumption intensity. Using it
as an organic demand-growth rate would assume the supply constraint persists, and
then report modest capacity requirements *because* demand was assumed not to grow —
which is circular. Same error class as the superseded 23.08 TWh demand base: a
quantity produced by the system's dysfunction read as a statement about underlying
demand.

The falling per-connected-person figure is itself the mechanism: households and
firms were connected to a grid that could not supply them, and the self-generation
economy absorbed the shortfall. That is the Tier 1 → Tier 2 gap, observed directly.

The historical series retains a legitimate role: it defines a
**`constrained_continuation`** arm — "what if the constraints persist" — as the
pessimistic bound, never the central case.

**External corroboration.** NIRP 2024 (Table 14) derives its national demand
forecast bottom-up from population, GDP, electrification rate and network-loss
trajectories, explicitly rather than from historical generation. The national
planning authority reached the same methodological conclusion independently.

---

## Values carried into `scenarios.py`

| Arm | Rate | Basis |
|---|---|---|
| `constrained_continuation` central | **3.14%** | Ember 2010–2025 generation CAGR |
| range floor | 1.53% | weakest window (2014–2024) |
| range ceiling | 3.22% | strongest window (2017–2025) |

Label as a **constraint-persistence** arm. These are not organic demand growth and
must never be presented as such.

---

## Sources

| Series | Source | Role |
|---|---|---|
| Grid generation, level | NERC quarterly reports, gross generation sent out at busbar | **Authoritative.** Sets the 37.09 TWh (2024) Tier 1 anchor |
| Grid generation, long series | Ember Electricity Data Explorer, Nigeria annual generation | **Shape only**, not level |
| Population | World Bank WDI `SP.POP.TOTL` (UN WPP + national statistical office) | Denominator |
| Access rate | World Bank WDI `EG.ELC.ACCS.ZS` | Connected-population denominator |

---

## Flags — read before using any number here

**[F1] Population figures are projections.** Nigeria's last census was 2006. The
projections are contested and the uncertainty propagates into every per-capita
figure.

**[F2] The NERC 2022 entry is an average-power figure** (3,988 MWh/h), not an annual
total. Converted as × 8,760 h. Whether it is sent-out or generated is unverified.

**[F3] The NERC-to-Ember gap is not a stable offset:** +8.8% (2022), +11.5% (2023),
+1.3% (2024), +6.0% (2025). A stable offset would permit index-splicing; this does
not. The two series are reported side by side and never merged.

**[F4] The access series is noisy and unit-inconsistent in the source.** 2010–2012
were transcribed as fractions (0.48, 0.559, 0.53) and 2013 onward as percentages;
normalised in the workbook. The series itself moves implausibly year to year (48% →
55.9% → 53%; 52.5% → 59.3% → 54.4%), reflecting survey revisions rather than real
connection events. **Single-year endpoints are unreliable; only the multi-window
direction should be relied on.** 2025 access is not yet published.

**[F5] WDI access includes off-grid and distributed renewables**, not solely grid
connections. "Connected population" therefore overstates *grid*-connected
population, increasingly so over time as DRE deployment grew. This means the
per-connected-person decline in Result 2 is **exaggerated in magnitude**, though not
in sign. The true intensity trend lies between Result 1 (+0.2 to +0.7%) and Result 2
(−0.8 to −2.7%) — in either case, at or below zero.

**[F6] The connected-basis windows end in 2024** because access data stops there,
and 2024 is a trough in the Ember series (37.56 TWh, against 40.92 in 2023 and 41.54
in 2025). This biases Result 2 downward. The 2010–2023 window (−0.82%/yr) is
included as the fair-minded alternative endpoint.

**[F7] 2026 Q1 (8,883.47 GWh) is a partial year** and is excluded.

---

## Further analysis

1. Complete the NERC annual series 2017–2022 from the quarterly reports, retiring
   dependence on Ember for shape.
2. If a *grid-connection* count is obtainable (NERC registered customers ×
   persons-per-connection), it would replace the WDI access denominator and close
   [F5]. Watch for administrative step-changes from the National Mass Metering
   Programme and MAP rollout, which move registered counts without moving
   connections.
