> **Provenance header (added 2026-09-20).** Compiled 2026-09-20. Figures
> in this file are **secondary citations transcribed from web sources**;
> **no primary document has been verified against the repo.** Treat every
> number as "as supplied, unchecked" until a primary is read.

# §3.5 SOURCING PACK 2 — ledger rows

Sourced 2026-09-20 by Chat. Three rows closed, three not. **Two of the
three closed carry a direction problem that must be checked before any
value is applied.**

---

## 1. Solar cost adder — CHECK THE SIGN BEFORE APPLYING

**Source: IRENA, *Renewable Power Generation Costs* (2025).**

| metric | value |
|---|---|
| Africa, utility-scale solar PV, total installed cost, 2025 | **USD 837/kW** |
| Same, 2015 | USD 3,520/kW (a 76% fall — the largest regional reduction of any region) |
| Africa, utility-scale solar LCOE, 2025 | **USD 52/MWh** |
| Oceania, same | USD 41/MWh |
| China's best projects | below USD 30/MWh |

**The problem.** The model applies a Nigeria/Africa *adder* on top of
NREL ATB. But ATB is a **US** cost basis, and US utility-scale PV
overnight capital cost sits well above USD 837/kW. On IRENA's numbers
Africa is **cheaper than the US in capex terms**, not dearer — so an
additive premium on ATB would be **double-counting**, and the adder may
be signed the wrong way.

Note also that Africa's LCOE premium (52 vs 41 USD/MWh) is **a cost-of-
capital effect, not a capex effect** — it is the same phenomenon
`r_commercial` already represents. Applying a capex adder *and* a high
private discount rate charges the same risk twice.

**Instruction:** report the adder's current value, sign, and what it is
applied to, and state the ATB baseline it modifies. **Change nothing.**

---

## 2. Self-generation energy — SOURCED

**Primary (IEA, 2019):** 40 TWh generated from 40 GW of backup capacity
across sub-Saharan Africa in 2018, of which **Nigeria accounted for
almost half — about 18 TWh from roughly 9 GW of backup generation.**
That is the citable energy figure.

**Share (IEA, reported January 2024):** backup generators cover about
**40% of Nigeria's electricity needs**.

**Capacity estimates, for the range:**

| estimate | source |
|---|---|
| ~13 GW (2.8 m residential sets, 210 k commercial) | IFC (2019), described as the most recent accurate estimate |
| ~14 GW small-scale diesel and petrol | Energy for Growth Hub / energypedia |
| 8–14 GW decentralised diesel | CTC-N |
| 15–20 GW private generators nationally | Lagos State-backed study, 2026 |

Corroborating: about 86% of Nigerian companies own or share a
generator, covering about 48% of their electricity demand; the Lagos
study finds 72% of households own at least one generator and 94% of
MSMEs.

**This row bears directly on the NIRP 2024 self-generation audit** —
the IEA 18 TWh / 40% figures are the independent check on NIRP's own
number.

---

## 3. Peak multiple — SOURCED, but the observable is NOT the parameter

**Observed grid data (NERC / TCN):**

| metric | value |
|---|---|
| All-time peak generation | **5,801.84 MW**, 4 March 2025, 49.69 Hz |
| Same-day maximum energy | 128,370.75 MWh → 5,348.8 MW average |
| **Implied peak multiple that day** | **1.085** |
| Grid load factor, 2026/Q1 | 92.26% (up 9.81 pp from 82.45% in 2025/Q4) |
| Grid load factor, April 2026 | 94% (4,048 MWh/h) |
| Grid load factor, January 2026 | 90% |
| Installed capacity | 13,625 MW |
| Average available capacity | 4,286 MW (Apr 2026, PAF 31%); 5,506 MW (Oct 2025, PAF 40%) |

**The trap.** A load factor of 90–94% and a peak multiple of ~1.09
would be extraordinary for any real demand profile. It is not a demand
observation — **it is an artefact of supply rationing.** The grid
dispatches nearly everything available at all hours because supply,
not demand, is binding; DisCos load-shed the rest. Estimated demand is
well above 20,000 MW against ~4,000 MW delivered.

So **the observed multiple is the *suppressed* value and must never be
used as the unsuppressed demand peak multiple.** Your `peak_case` arms
already encode this distinction (`peak_suppressed` /
`peak_unsuppressed`); this sourcing tells you what the suppressed end
of that range actually is, and confirms that any unsuppressed multiple
must come from a load-research study or a peer-system analogue, not
from Nigerian grid data.

**Instruction:** report what each `peak_case` arm currently uses and its
stated provenance. Log the observed figures above as the **suppressed**
anchor. **Change nothing.**

---

## 4–6. NOT CLOSED in this pass

- **Guarantee headroom** — needs DMO contingent-liability and guarantee
  data; not sourced.
- **Concessional envelope size** beyond the $750M DARES credit — the
  repo corroborates $750M; the $1,050M arm remains STIPULATED.
- **Fiscal-realisation rate** (the misnamed `disco_collection_rate`) —
  the right proxy is Nigeria's federal **capital-budget implementation
  rate**, from Budget Office / DMO releases; not sourced.

These three need a dedicated pass. None blocks the current task.

---

## 7. Standing position on §3.5

Unchanged: **the gate cannot close.** Blocker 2 (tariff grid against
T\*(t, blend)) is a computation, not a citation, and nothing in either
pack touches it.
