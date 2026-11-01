# Gas fleet retirement schedule — provenance memo

**Status: `ASSUMED`. Not traced. Read this before citing any result that
depends on `gas_retirement_case`.**

Companion to `data/capacity/raw/gas_plant_vintages.csv`.

---

## 1. What this replaces, and why

`gas_baseline_retirement_mw_per_year = 680.0` from 2035 was a bare literal
described in its own comment as a "brownfield retirement proxy". It retires the
entire 10,947 MW fleet by 2051 — three years before the horizon ends — and
hands the whole firm-capacity burden to storage.

Measured cost of that assumption (`peak_unsuppressed`, `storage_cc_central`,
`gas_none`, `gas_avail_nameplate`, sliced baseline):

| arm | objective | storage 2045 | binding years |
|---|---|---|---|
| `gas_retire_proxy` (680 MW/yr) | 22.873 bn | 96,735 MWh | 21/30 |
| `gas_retire_none` (0 MW/yr) | 19.278 bn | 52,947 MWh | 21/30 |
| **difference** | **3.60 bn (15.7%)** | **43,788 MWh (45%)** | unchanged |

So 45% of the storage build and 16% of system cost rest on an unsourced number.
Note that the binding count and every firm-capacity dual are identical across
the two arms — retirement changes the *quantity* of storage required, not the
*marginal price* of firm capacity, because storage is the marginal supplier in
both cases.

## 2. Provenance of the replacement — stated plainly

The plant-by-plant commissioning years in
`data/capacity/raw/gas_plant_vintages.csv` are an **analyst estimate produced
from general knowledge of Nigerian power-sector project histories**. They are
**not** traced to NERC, NBET, NDPHC, TCN, or company filings. Nigerian
commissioning-date records were not obtainable within the project timeframe.

Evidentiary status is therefore identical to NIRP's stipulated 27 TWh
self-generation figure and the λ ratio: a quantity used because no better one is
available, carried as a declared assumption rather than a citation.

**The arm must not be named `gas_retire_sourced`.** Use `gas_retire_vintage_30`
/ `_40` / `_50`. A parameter whose name asserts provenance it does not have is
the failure mode this project has already corrected twice.

Confidence by capacity, as flagged per row in the CSV:

| confidence | MW | share |
|---|---|---|
| H | 4,066 | 35.8% |
| M | 5,364 | 47.2% |
| L | 1,931 | 17.0% |

The largest single weakness is **Afam IV-V (726 MW, confidence L)**: Afam IV
dates from the 1970s and Afam V from around 2002, so the single 1990 value is a
blend across two vintages that should really be separate rows. If the schedule
is ever revisited, split that plant first.

## 3. Derived schedules

Only plants whose commissioning year plus assumed life falls inside 2025–2054
retire within the horizon.

| technical life | retires by 2054 | remaining 2054 | mean 2035–54 |
|---|---|---|---|
| 30 yr | 7,695 MW (68%) | 3,666 MW | 365 MW/yr |
| 40 yr | 8,780 MW (77%) | 2,581 MW | 292 MW/yr |
| 50 yr | 4,060 MW (36%) | 7,301 MW | 167 MW/yr |
| *proxy (for comparison)* | *10,947 MW (100%)* | *0 MW* | *680 MW/yr* |

**Every vintage arm retires far less than the proxy** — roughly half at 30-year
life, a quarter at 50. The proxy is therefore an upper bound on the retirement
artefact, and the true cost of the assumption sits between `gas_retire_none`
and the vintage arms, nearer to `none`.

## 4. Why a flat technical life is itself wrong, and which arm to prefer

The 40-year case retires Delta in 2025 and Egbin in 2026 — implausible on its
face, since both are operating. That exposes the real defect: **a flat life
measured from original commissioning does not describe a fleet that has been
rehabilitated.**

Two effects run in opposite directions:

- **Rehabilitation extends life beyond nameplate vintage.** Transcorp overhauled
  Delta after 2013; Egbin has been repowered. Measuring 30 years from 1985–86
  understates their remaining service.
- **Under-utilisation also extends calendar life**, because running hours rather
  than years drive thermal degradation. The NIPP fleet has barely operated —
  Alaoji reports ~0% availability in both NERC 2024 and 2026/Q1 — so its assets
  are chronologically old and mechanically young.

These partly cancel. **Take `gas_retire_vintage_30` as central** — not because
30 years is correct, but because the two errors offset and 30 gives the most
plausible mid-horizon trajectory. Carry 50 as the upper bound on fleet survival
and `gas_retire_none` as the absolute upper bound.

## 5. Interaction to watch

Under `gas_build_case = "gas_none"`, retirement and replacement are not
symmetric: the fleet declines and nothing can replace it. That combination is
not a scenario anyone would defend — it says Nigeria retires its thermal fleet
and builds nothing, while NDC 3.0 targets 17 GW of gas by 2035.

**Every headline result must therefore pair a retirement arm with a build arm.**
The policy-relevant pairing is `gas_retire_vintage_30` + `gas_ndc_17gw`, which
is currently unrunnable pending the CCGT capex and efficiency parameters.
Results produced under `gas_none` are diagnostic artefacts, not pathways.

## 6. What would resolve this

Commissioning dates from any of: NERC quarterly reports (best — same source as
Table 4.3, so the basis would match exactly); NDPHC project completion records
for the ten NIPP plants; annual reports for the listed and privatised assets
(Geregu Power Plc, Transcorp, Sahara/KEPCO, Azura-Edo, NAOC).

If any subset becomes available, replace those rows and re-flag their confidence
to H. The CSV is structured so partial replacement is possible without redoing
the whole table.
